import gc
import warnings

import numpy as np
import polars as pl
import torch
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.tab_model import TabNetRegressor

import wandb
from config import cfg
from data.data_class import DatasetXy

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


def train_model(fold: int, datasetXy: DatasetXy):
    X_train, y_train, X_valid, y_valid, X_test, y_test = datasetXy.get()

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

    # Reshape X
    X_train = X_train.to_numpy()
    X_valid = X_valid.to_numpy()

    # Reshape y
    y_train = y_train.to_numpy().reshape(-1, 1) if hasattr(y_train, "to_numpy") else np.array(y_train).reshape(-1, 1)
    y_valid = y_valid.to_numpy().reshape(-1, 1) if hasattr(y_valid, "to_numpy") else np.array(y_valid).reshape(-1, 1)

    # Set up model with proper categorical indices and dimensions
    tabnet_params = {
        # Architecture parameters
        "n_d": 8,  # Width of the decision prediction layer (increased from default 8)
        "n_a": 8,  # Width of the attention embedding for each step (increased from default 8)
        "n_steps": 5,  # Number of steps in the architecture (increased from default 3)
        "gamma": 1.5,  # Coefficient for feature reusage in the masks
        # For categorical features from your feature engineering
        "cat_idxs": cat_cols_idx,
        "cat_dims": cat_dims,
        "cat_emb_dim": 3,  # Embedding dimension for categorical features (increased slightly)
        # Feature selection parameters
        "n_independent": 2,  # Number of independent Gated Linear Units layers
        "n_shared": 3,  # Number of shared Gated Linear Units (increased from default)
        # Regularization parameters
        "lambda_sparse": 0.005,  # Sparsity regularization (increased slightly)
        "momentum": 0.3,  # Ghost Batch Norm momentum (increased)
        "clip_value": 2,  # Gradient clipping value (increased)
        # Training parameters
        "optimizer_fn": torch.optim.AdamW,
        "optimizer_params": {
            "lr": 0.01,
            "weight_decay": 1e-5,
        },
        # Learning rate scheduler
        "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 10, "verbose": True},
        # Other parameters
        "mask_type": "entmax",
        "verbose": 1,
        "seed": cfg.random_state,
        "device_name": cfg.device,
    }
    model = TabNetRegressor(**tabnet_params)

    # Train the model with our proper callback class
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=["valid"],
        eval_metric=["rmse"],
        max_epochs=500,
        patience=3,
        batch_size=1024 * 10,
        virtual_batch_size=128 * 10,
        callbacks=[WandbCallback()],
    )

    del X_train, y_train, X_valid, y_valid
    gc.collect()
    torch.cuda.empty_cache()

    # Get validation score
    val_score = min(model.history["valid_rmse"])
    print(f"Validation score: {val_score}")
    wandb.summary["best_val_score"] = val_score

    # feature_importances = model.feature_importances_
    # importance_dict = {col: imp for col, imp in zip(X_train.columns, feature_importances)}

    # # Log feature importances as a wandb Table
    # feature_importance_table = wandb.Table(columns=["Feature", "Importance"])
    # for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
    #     feature_importance_table.add_data(feature, importance)

    # # Log the table and also as a bar chart for visualization
    # wandb.log({"feature_importances": feature_importance_table})
    # wandb.log({"feature_importance_plot": wandb.plot.bar(feature_importance_table, "Feature", "Importance", title="Feature Importances")})

    # # Also log as simple key-value pairs for easy access
    # wandb.log({"importance/" + feature: importance for feature, importance in importance_dict.items()})

    if hasattr(cfg, "predict") and cfg.predict:
        for col, mapping in category_mappings.items():
            X_test = X_test.with_columns(pl.col(col).map_elements(lambda x: mapping.get(x, None)).alias(col))
        X_test = X_test.to_numpy()
        y_test = model.predict(X_test)

        del X_test
        gc.collect()
        torch.cuda.empty_cache()

        return val_score, y_test.flatten().tolist()

    return val_score, None
