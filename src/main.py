import gc
import os
import warnings
from dataclasses import asdict

import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

import wandb
from config import cfg
from data_process import get_dfs
from utils import WandbCallback, commit_results

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb_run = wandb.init(project="playground-series-s5e4", config=asdict(cfg))

df_train, df_test, df_sub, y_train, df_desc, before_encode_len = get_dfs()
print(df_train.columns)
print(df_train)
print(df_desc)

X = df_train.copy()
y = y_train.copy()
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% for validation
    random_state=42,
)

X_test = df_test[X.columns].copy()

# # Target encoding if needed
# encoded_columns = df_train.columns[cfg.encoded_columns_start:]
# encoder = TargetEncoder(random_state=cfg.random_state)

# X_train[encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
# X_valid[encoded_columns] = encoder.transform(X_valid[encoded_columns])
# X_test[encoded_columns] = encoder.transform(X_test[encoded_columns])

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
