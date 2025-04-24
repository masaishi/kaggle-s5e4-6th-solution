import gc

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

import wandb
from config import cfg
from data.data_class import DatasetXy


class WandbCallback:
    def __init__(self, log_every=50, log_feature_importance=True):
        self.log_every = log_every
        self.iteration = 0
        self.log_feature_importance = log_feature_importance
        self.model = None
        self.best_score = float("inf")
        self.best_iteration = 0
        self.early_stopping_rounds = 150
        self.no_improvement_count = 0

    def update(self, model, X_train, y_train, X_valid, y_valid, iteration):
        self.iteration = iteration

        # Calculate metrics
        train_pred = model.predict(X_train)
        valid_pred = model.predict(X_valid)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

        metrics = {"train/rmse": train_rmse, "valid/rmse": valid_rmse}

        # Check for early stopping
        if valid_rmse < self.best_score:
            self.best_score = valid_rmse
            self.best_iteration = iteration
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Log feature importance if available
        if self.log_feature_importance and hasattr(model, "coef_"):
            feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]
            importance = np.abs(model.coef_[0])
            feature_importance = {name: float(imp) for name, imp in zip(feature_names, importance)}

            wandb.log(
                {
                    "feature_importance": wandb.Table(
                        data=[[k, v] for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)],
                        columns=["feature", "importance"],
                    )
                },
                step=iteration,
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
                step=iteration,
            )

        # Log metrics
        wandb.log(metrics, step=iteration)

        # Return True if early stopping should occur
        return self.no_improvement_count >= self.early_stopping_rounds


def train_model(fold: int, datasetXy: DatasetXy):
    X_train, y_train, X_valid, y_valid, X_test, y_test = datasetXy.get()

    X_train = X_train.to_pandas()
    y_train = y_train.to_pandas()
    X_valid = X_valid.to_pandas()
    y_valid = y_valid.to_pandas()

    wandb_callback = WandbCallback(log_every=50)

    # Configure SVM hyperparameters (can be moved to config)
    svm_params = {
        "kernel": "rbf",  # Radial basis function kernel
        "C": 1.0,  # Regularization parameter
        "epsilon": 0.1,  # Epsilon in the epsilon-SVR model
        "gamma": "scale",  # Kernel coefficient
        "verbose": True,
        "max_iter": 1000,  # Maximum number of iterations
    }

    model = SVR(**svm_params)

    best_model = None
    max_iterations = 10  # Number of hyperparameter tuning iterations

    for iteration in range(max_iterations):
        # Adjust C parameter to simulate iterative training
        model.set_params(C=svm_params["C"] * (1 + iteration * 0.1))

        # Fit the model
        model.fit(X_train, y_train)

        # Update wandb with metrics and check for early stopping
        should_stop = wandb_callback.update(model, X_train, y_train, X_valid, y_valid, iteration)

        if iteration == 0 or wandb_callback.best_iteration == iteration:
            best_model = model

        if should_stop:
            print(f"Early stopping at iteration {iteration}")
            break

    # Use the best model
    model = best_model

    del X_train, y_train, X_valid, y_valid
    gc.collect()

    val_score = wandb_callback.best_score
    print(f"Fold {fold + 1} validation score: {val_score}")
    wandb.log({f"fold_{fold + 1}_val_score": val_score})

    if hasattr(cfg, "predict") and cfg.predict:
        X_test = X_test.to_pandas()
        y_test = model.predict(X_test)
        return val_score, y_test.tolist()

    return val_score, None
