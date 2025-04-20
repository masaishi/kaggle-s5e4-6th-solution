# tabnet_regressor.py
import gc
import os
import warnings

import numpy as np
import torch
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import OrdinalEncoder

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

    # Identify categorical columns
    cat_cols_idx = [i for i, dtype in enumerate(X_train.dtypes) if dtype == "object" or dtype == "category"]

    # Convert DataFrames to numpy for processing
    X_train_values = X_train.values.copy()
    X_valid_values = X_valid.values.copy()

    # Process categorical features with scikit-learn
    if cat_cols_idx:
        encoders = {}
        for idx in cat_cols_idx:
            # Create and fit encoder for each categorical column
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train_values[:, idx] = encoder.fit_transform(X_train_values[:, idx].reshape(-1, 1)).flatten()
            X_valid_values[:, idx] = encoder.transform(X_valid_values[:, idx].reshape(-1, 1)).flatten()
            encoders[idx] = encoder

    # Reshape target variables to 2D as required by TabNetRegressor
    y_train_values = y_train.values.reshape(-1, 1) if hasattr(y_train, "values") else np.array(y_train).reshape(-1, 1)
    y_valid_values = y_valid.values.reshape(-1, 1) if hasattr(y_valid, "values") else np.array(y_valid).reshape(-1, 1)
    print(X_train_values.shape, y_train_values.shape, X_valid_values.shape, y_valid_values.shape)

    # Initialize wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = {"learning_rate": 2e-2, "n_iter": 200, "early_stopping": 10, "metric": "rmse", "n_d": 64, "n_a": 64, "n_steps": 5}
    wandb_run = wandb.init(project="playground-series-s5e4", config=config)

    # Since we've pre-encoded the categorical features, we don't need TabNet to handle them
    # We'll pass empty lists for categorical indices and dimensions
    model = TabNetRegressor(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        cat_idxs=[],  # We've already encoded categoricals
        cat_dims=[],  # We've already encoded categoricals
        optimizer_fn=torch.optim.Adam,
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
