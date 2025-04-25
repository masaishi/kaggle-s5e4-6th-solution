import gc

import lightgbm as lgb

import wandb
from config import cfg
from data.data_class import DatasetXy


class WandbCallback:
    def __init__(self, log_every=50, log_feature_importance=True):
        self.log_every = log_every
        self.iteration = 0
        self.log_feature_importance = log_feature_importance

    def __call__(self, env):
        if self.iteration % self.log_every == 0:
            metrics = {}
            for dataset_name, eval_name, value, _ in env.evaluation_result_list:
                metric_name = f"{dataset_name}/{eval_name}"
                metrics[metric_name] = value

            if self.log_feature_importance and hasattr(env, "model"):
                feature_names = env.model.feature_name()
                importance = env.model.feature_importance(importance_type="split")
                feature_importance = {name: imp for name, imp in zip(feature_names, importance)}

                wandb.log(
                    {
                        "feature_importance": wandb.Table(
                            data=[[k, v] for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)],
                            columns=["feature", "importance"],
                        )
                    },
                    step=self.iteration,
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
                    step=self.iteration,
                )

            wandb.log(metrics, step=self.iteration)

        self.iteration += 1
        return False


def train_model(fold: int, datasetXy: DatasetXy):
    X_train, y_train, X_valid, y_valid, X_test, y_test = datasetXy.get()

    X_train = X_train.to_pandas()
    y_train = y_train.to_pandas()
    X_valid = X_valid.to_pandas()
    y_valid = y_valid.to_pandas()

    model = lgb.LGBMRegressor(
        n_iter=cfg.n_iter,
        max_depth=cfg.max_depth,
        num_leaves=cfg.num_leaves,
        colsample_bytree=cfg.colsample_bytree,
        learning_rate=cfg.learning_rate,
        objective=cfg.objective,
        metric=cfg.metric,
        verbosity=cfg.verbosity,
        random_state=cfg.random_state,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        callbacks=[
            lgb.log_evaluation(cfg.log_eval),
            lgb.early_stopping(cfg.early_stopping),
            WandbCallback(log_every=50),
        ],
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
