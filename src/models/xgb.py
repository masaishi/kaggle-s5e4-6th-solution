import gc
import os
import warnings
from dataclasses import asdict

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

import wandb
from config import cfg
from data_class import Dfs
from data_process import add_fold, get_Xy
from utils import commit_results

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


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


load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb_run = wandb.init(project="playground-series-s5e4", config=asdict(cfg))

if hasattr(cfg, "eval") and cfg.eval:
    cfg.n_iter = 500
if hasattr(cfg, "debug") and cfg.debug:
    cfg.n_iter = 5

df = pl.read_csv(cfg.train_path)
df = df.filter(pl.col("Number_of_Ads").is_not_null())

df_test = None
if hasattr(cfg, "predict") and cfg.predict:
    df_test = pl.read_csv(cfg.test_path)

df = add_fold(df)

test_preds = []
group_kfold = GroupKFold(n_splits=5)
for fold, (idx_train, idx_valid) in enumerate(group_kfold.split(df, groups=df["fold"])):
    df_train = df[idx_train]
    df_valid = df[idx_valid]

    datasetXy = get_Xy(Dfs(df_train=df_train, df_valid=df_valid, df_test=df_test))
    X_train, y_train, X_valid, y_valid, X_test, y_test = datasetXy.get()

    X_train = X_train.to_pandas()
    y_train = y_train.to_pandas()
    X_valid = X_valid.to_pandas()
    y_valid = y_valid.to_pandas()

    # Create XGBoost callback
    wandb_callback = WandbCallback(log_every=50)

    # XGBoost model instead of LightGBM
    # Note: XGBoost uses 'reg:squarederror' for regression objectives
    model = XGBRegressor(
        n_estimators=cfg.n_iter,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        objective="reg:squarederror",  # XGBoost regression objective
        colsample_bytree=cfg.colsample_bytree,
        random_state=42,
        verbosity=cfg.verbosity,
    )

    # Store model in callback for feature importance logging
    wandb_callback.model = model

    # Fit the XGBoost model - note we use eval_metric parameter within a list called 'callbacks'
    # and we add early_stopping_rounds as a separate parameter
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=cfg.log_eval,
        callbacks=[wandb_callback],
        early_stopping_rounds=cfg.early_stopping,
        eval_metric=cfg.metric,  # This should now be passed directly
    )

    del X_train, y_train, X_valid, y_valid
    gc.collect()

    if hasattr(cfg, "predict") and cfg.predict:
        X_test = X_test.to_pandas()
        y_test = model.predict(X_test)
        test_preds += [y_test.tolist()]

    # Get validation score from model
    val_score = model.best_score
    print(f"Fold {fold + 1} validation score: {val_score}")
    wandb.log({f"fold_{fold + 1}_val_score": val_score})

    gc.collect()
    if hasattr(cfg, "eval") and cfg.eval and fold >= 0:
        break

git_info = commit_results(val_score, wandb_run.name)
wandb.config.update(git_info)
wandb.finish()

gc.collect()

if hasattr(cfg, "predict") and cfg.predict:
    test_pred = np.array(test_preds).mean(axis=0)

    test_df = pl.read_csv(cfg.test_path)
    test_df = test_df.with_columns(pl.Series(test_pred).alias("Listening_Time_minutes"))
    test_df = test_df[["id", "Listening_Time_minutes"]]
    wandb_num = wandb_run.name.split("-")[-1]
    test_df.write_csv(f"./data/submissions/sub-{wandb_num}.csv")
    print(f"Test predictions saved to ./data/submissions/sub-{wandb_num}.csv")
