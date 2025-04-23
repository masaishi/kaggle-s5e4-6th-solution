import gc
import os
import warnings
from dataclasses import asdict

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold

import wandb
from config import cfg
from data.data_class import Dfs
from data.data_process import add_fold, get_Xy
from models.lgb import train_model

# from models.xgb import train_model
from utils import commit_results

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


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

val_score = 999
test_preds = []
group_kfold = GroupKFold(n_splits=cfg.num_fold)
for fold, (idx_train, idx_valid) in enumerate(group_kfold.split(df, groups=df["fold"])):
    df_train = df[idx_train]
    df_valid = df[idx_valid]

    datasetXy = get_Xy(Dfs(df_train=df_train, df_valid=df_valid, df_test=df_test))

    val_score, test_pred = train_model(fold, datasetXy)
    test_preds += [test_pred]

    gc.collect()
    if hasattr(cfg, "eval") and cfg.eval and fold >= 1:
        break

git_info = commit_results(val_score, wandb_run.name)
wandb.config.update(git_info)
wandb.finish()

gc.collect()

if hasattr(cfg, "predict") and cfg.predict:
    breakpoint()
    test_pred = np.array(test_preds).mean(axis=0)

    test_df = pl.read_csv(cfg.test_path)
    test_df = test_df.with_columns(pl.Series(test_pred).alias("Listening_Time_minutes"))
    test_df = test_df[["id", "Listening_Time_minutes"]]
    wandb_num = wandb_run.name.split("-")[-1]
    test_df.write_csv(f"./data/submissions/sub-{wandb_num}.csv")
    print(f"Test predictions saved to ./data/submissions/sub-{wandb_num}.csv")
