import gc
import os
import warnings

import numpy as np
import torch
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder

import wandb
from data_process import get_dfs
from utils import commit_results

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


# Create a proper callback class
class WandbCallback(Callback):
    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            wandb.log({metric_name: metric_value})
        return False


def train_tabnet_model():
    # Load data
    dfs = get_dfs()
    X_train, y_train, X_valid, y_valid, X_test = dfs.values()
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape)
    print(X_train.head())

    # Identify categorical columns
    cat_cols = [col for col in X_train.columns if X_train[col].dtype == "object" or X_train[col].dtype == "category"]
    cat_cols_idx = [i for i, col in enumerate(X_train.columns) if col in cat_cols]

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on train and transform train, valid, and test
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_valid[col] = le.transform(X_valid[col].astype(str))
        if col in X_test.columns:  # Ensure column exists in test data
            X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    # Convert DataFrames to numpy for processing
    X_train_values = X_train.values.copy()
    X_valid_values = X_valid.values.copy()

    # Reshape target variables to 2D as required by TabNetRegressor
    y_train_values = y_train.values.reshape(-1, 1) if hasattr(y_train, "values") else np.array(y_train).reshape(-1, 1)
    y_valid_values = y_valid.values.reshape(-1, 1) if hasattr(y_valid, "values") else np.array(y_valid).reshape(-1, 1)
    print(X_train_values.shape, y_train_values.shape, X_valid_values.shape, y_valid_values.shape)

    # Initialize wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = {"learning_rate": 2e-2, "n_iter": 200, "early_stopping": 10, "metric": "rmse", "n_d": 64, "n_a": 64, "n_steps": 5}
    wandb_run = wandb.init(project="playground-series-s5e4", config=config)

    # Now that we've encoded the categorical features properly, we can pass them to TabNet
    # Calculate cat_dims from the encoded data (number of unique values for each categorical feature)
    cat_dims = []
    for i, col in enumerate(X_train.columns):
        if i in cat_cols_idx:
            cat_dims.append(len(label_encoders[col].classes_))

    # Set up model with proper categorical indices and dimensions
    model = TabNetRegressor(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        cat_idxs=cat_cols_idx,
        cat_dims=cat_dims,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={"lr": 2e-2},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type="sparsemax",
        lambda_sparse=1e-3,
        seed=42,
        verbose=1,
    )

    # Train the model with our proper callback class
    model.fit(
        X_train=X_train_values,
        y_train=y_train_values,
        eval_set=[(X_valid_values, y_valid_values)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=200,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        callbacks=[WandbCallback()],
    )

    # Get validation score
    val_score = min(model.history["valid_rmse"])
    print(f"Validation score: {val_score}")
    wandb.summary["best_val_score"] = val_score

    # Log results and clean up
    git_info = commit_results(val_score, wandb_run.name)
    wandb.config.update(git_info)
    wandb.finish()
    gc.collect()

    return model


if __name__ == "__main__":
    model = train_tabnet_model()
    print("Model training completed.")
