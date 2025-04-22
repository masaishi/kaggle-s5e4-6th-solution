import gc
import os
import warnings

import numpy as np
import polars as pl
import torch
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetRegressor

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
    X_train, y_train, X_valid, y_valid = dfs.values()
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    print(X_train.head())

    # Handle categorical columns
    cat_cols = [col for col in X_train.columns if pl.Categorical in X_train[col].dtype.base_type().__mro__ or X_train[col].dtype == pl.Utf8]
    cat_cols_idx = [i for i, col in enumerate(X_train.columns) if col in cat_cols]
    category_mappings = {}
    cat_dims = []

    combined_data = pl.concat([X_train, X_valid], how="vertical")
    for col in cat_cols:
        cat_dims.append(combined_data[col].unique().count())

        unique_values = combined_data[col].unique().sort()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        category_mappings[col] = mapping

        X_train = X_train.with_columns(pl.col(col).map_elements(lambda x: mapping.get(x, None)).alias(col))
        X_valid = X_valid.with_columns(pl.col(col).map_elements(lambda x: mapping.get(x, None)).alias(col))

    # Convert Polars DataFrames to numpy arrays
    X_train_np = X_train.to_numpy()
    X_valid_np = X_valid.to_numpy()

    # Reshape target variables
    y_train_np = y_train.to_numpy().reshape(-1, 1) if hasattr(y_train, "to_numpy") else np.array(y_train).reshape(-1, 1)
    y_valid_np = y_valid.to_numpy().reshape(-1, 1) if hasattr(y_valid, "to_numpy") else np.array(y_valid).reshape(-1, 1)

    print(X_train_np.shape, y_train_np.shape, X_valid_np.shape, y_valid_np.shape)

    # Initialize wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = {"learning_rate": 2e-2, "n_iter": 200, "early_stopping": 10, "metric": "rmse", "n_d": 64, "n_a": 64, "n_steps": 5}
    wandb_run = wandb.init(project="playground-series-s5e4", config=config)

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
        X_train=X_train_np,
        y_train=y_train_np,
        eval_set=[(X_train_np, y_train_np), (X_valid_np, y_valid_np)],
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

    feature_importances = model.feature_importances_
    importance_dict = {col: imp for col, imp in zip(X_train.columns, feature_importances)}

    # Log feature importances as a wandb Table
    feature_importance_table = wandb.Table(columns=["Feature", "Importance"])
    for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        feature_importance_table.add_data(feature, importance)

    # Log the table and also as a bar chart for visualization
    wandb.log({"feature_importances": feature_importance_table})
    wandb.log({"feature_importance_plot": wandb.plot.bar(feature_importance_table, "Feature", "Importance", title="Feature Importances")})

    # Also log as simple key-value pairs for easy access
    wandb.log({"importance/" + feature: importance for feature, importance in importance_dict.items()})

    # Log results and clean up
    git_info = commit_results(val_score, wandb_run.name)
    wandb.config.update(git_info)
    wandb.finish()
    gc.collect()

    return model


if __name__ == "__main__":
    model = train_tabnet_model()
    print("Model training completed.")
