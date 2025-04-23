import gc
import os
import warnings
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold

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


load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb_run = wandb.init(project="playground-series-s5e4", config=asdict(cfg))

if hasattr(cfg, "eval") and cfg.eval:
    cfg.n_iter = 500

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

    model = lgb.LGBMRegressor(
        n_iter=cfg.n_iter,
        max_depth=cfg.max_depth,
        num_leaves=cfg.num_leaves,
        colsample_bytree=cfg.colsample_bytree,
        learning_rate=cfg.learning_rate,
        objective=cfg.objective,
        metric=cfg.metric,
        verbosity=cfg.verbosity,
        random_state=42,
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

    if hasattr(cfg, "predict") and cfg.predict and X_test is not None:
        y_test = model.predict(X_test)
        test_preds += [y_test.to_list()]

    val_score = model.best_score_["valid_1"][cfg.metric]
    print(f"Fold {fold + 1} validation score: {val_score}")
    wandb.log({f"fold_{fold + 1}_val_score": val_score})

    if hasattr(cfg, "eval") and cfg.eval and fold >= 0:
        break

git_info = commit_results(val_score, wandb_run.name)
wandb.config.update(git_info)
wandb.finish()

gc.collect()

if hasattr(cfg, "predict") and cfg.predict and X_test is not None:
    test_pred = np.array(test_preds).mean(axis=0)

    test_df = pl.read_csv(cfg.test_path)
    test_df = test_df.with_columns(pl.Series(test_pred).alias("Listening_Time_minutes"))
    test_df.write_csv(cfg.test_output_path, has_header=True)
    print(f"Test predictions saved to {cfg.test_output_path}")
