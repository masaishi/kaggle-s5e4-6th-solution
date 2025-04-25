import gc
import os
import warnings
from math import sqrt

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import wandb
from data.data_class import DatasetXy

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def train_model(fold: int, datasetXy: DatasetXy):
    """
    Train a HistGradientBoostingRegressor model using the provided dataset.

    Args:
        fold: Current fold number
        datasetXy: Dataset class containing train, validation, and test data

    Returns:
        model: Trained model
        y_test: Predictions on test set if required, otherwise None
    """
    # Get data from dataset class
    X_train, y_train, X_valid, y_valid, X_test, y_test = datasetXy.get()

    # Convert polars dataframes to pandas
    X_train = X_train.to_pandas()
    y_train = y_train.to_pandas() if hasattr(y_train, "to_pandas") else pd.Series(y_train)
    X_valid = X_valid.to_pandas()
    y_valid = y_valid.to_pandas() if hasattr(y_valid, "to_pandas") else pd.Series(y_valid)

    print(f"Training data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Validation data shapes: X_valid={X_valid.shape}, y_valid={y_valid.shape}")

    # Initialize wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = {
        "learning_rate": 0.1,
        "max_iter": 1000,
        "max_depth": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "max_bins": 255,
        "early_stopping": 10,
        "metric": "rmse",
    }
    wandb_run = wandb.init(project="playground-series-s5e4", config=config)

    # Initialize model
    model = HistGradientBoostingRegressor(
        learning_rate=config["learning_rate"],
        max_iter=config["max_iter"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        l2_regularization=config["l2_regularization"],
        max_bins=config["max_bins"],
        random_state=42,
        verbose=1,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = model.predict(X_valid)
    val_score = sqrt(mean_squared_error(y_valid, val_preds))
    print(f"Fold {fold + 1} validation score: {val_score}")

    # Log final validation score
    wandb.log({f"fold_{fold + 1}_val_score": val_score})
    wandb.summary["best_val_score"] = val_score

    # Make predictions on test set if required
    if X_test is not None:
        X_test = X_test.to_pandas()
        y_pred = model.predict(X_test)

        # Clean up memory
        del X_train, y_train, X_valid, y_valid
        gc.collect()

        return val_score, y_pred.tolist()

    # Clean up memory
    del X_train, y_train, X_valid, y_valid
    gc.collect()

    return val_score, None
