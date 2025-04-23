import gc

from xgboost import XGBRegressor

import wandb
from config import cfg
from data.data_class import DatasetXy


class WandbCallback:
    def __init__(self, log_every=50, log_feature_importance=True):
        self.log_every = log_every
        self.iteration = 0
        self.log_feature_importance = log_feature_importance
        self.model = None

    def __call__(self, env):
        # In XGBoost callback, env contains information about the current iteration
        iteration = env.iteration

        if iteration % self.log_every == 0:
            metrics = {}
            # XGBoost callback provides evaluation results differently
            for dataset_name, eval_res in zip(["train", "valid"], env.evaluation_result_list):
                eval_name, value, _ = eval_res
                metric_name = f"{dataset_name}/{eval_name}"
                metrics[metric_name] = value

            if self.log_feature_importance and self.model is not None:
                # Get feature importance for XGBoost
                feature_names = self.model.get_booster().feature_names
                importance = self.model.get_booster().get_score(importance_type="weight")
                feature_importance = {name: float(importance.get(name, 0)) for name in feature_names}

                wandb.log(
                    {
                        "feature_importance": wandb.Table(
                            data=[[k, v] for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)],
                            columns=["feature", "importance"],
                        )
                    },
                    step=iteration,
                )

                wandb.log(
                    {
                        "feature_importance_plot": wandb.plot.bar(
                            wandb.Table(
                                data=[[k, v] for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]],
                                columns=["feature", "importance"],
                            ),
                            "feature",
                            "importance",
                            title="Top Feature Importance",
                        )
                    },
                    step=iteration,
                )

            wandb.log(metrics, step=iteration)

        return False


def train_model(fold: int, datasetXy: DatasetXy):
    X_train, y_train, X_valid, y_valid, X_test, y_test = datasetXy.get()

    X_train = X_train.to_pandas()
    y_train = y_train.to_pandas()
    X_valid = X_valid.to_pandas()
    y_valid = y_valid.to_pandas()

    wandb_callback = WandbCallback(log_every=50)

    model = XGBRegressor(
        n_estimators=cfg.n_iter,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        objective="reg:squarederror",
        colsample_bytree=cfg.colsample_bytree,
        random_state=42,
        verbosity=cfg.verbosity,
        callbacks=[wandb_callback],
    )

    wandb_callback.model = model

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=cfg.log_eval,
        early_stopping_rounds=cfg.early_stopping,
        eval_metric=cfg.metric,
    )

    del X_train, y_train, X_valid, y_valid
    gc.collect()

    val_score = model.best_score_["valid_1"][cfg.metric]
    print(f"Fold {fold + 1} validation score: {val_score}")
    wandb.log({f"fold_{fold + 1}_val_score": val_score})

    if hasattr(cfg, "predict") and cfg.predict:
        X_test = X_test.to_pandas()
        y_test = model.predict(X_test)
        return val_score, y_test.tolist()

    return val_score, None
