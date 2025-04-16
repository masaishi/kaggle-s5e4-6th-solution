import gc
import os
import warnings
from dataclasses import asdict

import lightgbm as lgb
from dotenv import load_dotenv

import wandb
from config import cfg
from data_process import get_dfs
from utils import WandbCallback, commit_results

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb_run = wandb.init(project="playground-series-s5e4", config=asdict(cfg))

dfs = get_dfs()
X_train, y_train, X_valid, y_valid, X_test = dfs.values()
print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape)

# Initialize the model
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

# Train model with validation
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

# Calculate validation score
val_score = model.best_score_["valid_1"][cfg.metric]
print(f"Validation score: {val_score}")

# AFTER validation score, commit the results and get commit info
git_info = commit_results(val_score, wandb_run.name)
wandb.config.update(git_info)

wandb.finish()
gc.collect()
