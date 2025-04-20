# tabnet_regressor_model.py
import gc
import os
import warnings
from dataclasses import asdict, dataclass

import torch
from dotenv import load_dotenv
from pytorch_tabnet.tab_model import TabNetRegressor

import wandb
from config import cfg
from data_process import get_dfs
from utils import commit_results

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
load_dotenv()


# Update config with TabNet specific parameters
@dataclass
class TabNetConfig:
    # Original config parameters
    learning_rate: float = getattr(cfg, "learning_rate", 2e-2)
    n_iter: int = getattr(cfg, "n_iter", 200)
    early_stopping: int = getattr(cfg, "early_stopping", 10)
    metric: str = "rmse"  # Explicitly set to rmse as requested
    verbosity: int = getattr(cfg, "verbosity", 1)

    # TabNet specific parameters
    n_d: int = 64  # Width of the decision prediction layer
    n_a: int = 64  # Width of the attention embedding
    n_steps: int = 5  # Number of steps in the architecture
    gamma: float = 1.5  # Coefficient for feature reusage
    lambda_sparse: float = 1e-3  # Sparsity coefficient
    batch_size: int = 1024
    virtual_batch_size: int = 128
    momentum: float = 0.02
    mask_type: str = "sparsemax"  # or "entmax"

    # Empty lists for categorical features (update if needed)
    cat_idxs: list = None
    cat_dims: list = None


# Enhanced WandbCallback for TabNet that logs both training and validation metrics
class EnhancedTabNetWandbCallback:
    def __init__(self, log_every=50):
        self.log_every = log_every
        self.counter = 0
        self.best_val_loss = float("inf")

    def __call__(self, step, loss, y_pred, y_true, model):
        self.counter += 1
        if self.counter % self.log_every == 0:
            # Log training loss
            wandb.log({"train_loss": loss})

            # Log validation metrics if available in model history
            if hasattr(model, "history") and model.history:
                # Get the most recent validation metrics
                if "val_rmse" in model.history and len(model.history["val_rmse"]) > 0:
                    val_rmse = model.history["val_rmse"][-1]
                    wandb.log({"val_rmse": val_rmse})

                # Log validation loss if available
                if "val_loss" in model.history and len(model.history["val_loss"]) > 0:
                    val_loss = model.history["val_loss"][-1]
                    wandb.log({"val_loss": val_loss})

                    # Track best validation loss
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        wandb.log({"best_val_loss": val_loss})

        return False  # Don't stop training


def train_tabnet_model():
    # Create new config
    tabnet_cfg = TabNetConfig()

    # Update wandb with the new config
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_run = wandb.init(project="playground-series-s5e4", config=asdict(tabnet_cfg))

    dfs = get_dfs()
    X_train, y_train, X_valid, y_valid, X_test = dfs.values()
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape)

    # Initialize the TabNetRegressor model with updated config
    model = TabNetRegressor(
        n_d=tabnet_cfg.n_d,
        n_a=tabnet_cfg.n_a,
        n_steps=tabnet_cfg.n_steps,
        gamma=tabnet_cfg.gamma,
        cat_idxs=tabnet_cfg.cat_idxs if tabnet_cfg.cat_idxs else [],
        cat_dims=tabnet_cfg.cat_dims if tabnet_cfg.cat_dims else [],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": tabnet_cfg.learning_rate},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type=tabnet_cfg.mask_type,
        momentum=tabnet_cfg.momentum,
        lambda_sparse=tabnet_cfg.lambda_sparse,
        seed=42,
        verbose=tabnet_cfg.verbosity,
    )

    # Train model with validation
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=["rmse"],  # Explicitly set to rmse
        max_epochs=tabnet_cfg.n_iter,
        patience=tabnet_cfg.early_stopping,
        batch_size=tabnet_cfg.batch_size,
        virtual_batch_size=tabnet_cfg.virtual_batch_size,
        callbacks=[EnhancedTabNetWandbCallback(log_every=50)],
    )

    # Calculate validation score
    val_score = min(model.history["val_rmse"])  # Explicitly use 'val_rmse'
    print(f"Validation score: {val_score}")
    wandb.summary["best_val_score"] = val_score
    wandb.log({"best_val_score": val_score})

    # Also log the best validation loss from the history
    if "val_loss" in model.history and len(model.history["val_loss"]) > 0:
        best_val_loss = min(model.history["val_loss"])
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.log({"final_best_val_loss": best_val_loss})

    # After validation score, commit the results and get commit info
    git_info = commit_results(val_score, wandb_run.name)
    wandb.config.update(git_info)
    wandb.finish()
    gc.collect()

    return model


if __name__ == "__main__":
    model = train_tabnet_model()
    print("Model training completed and predictions generated.")
