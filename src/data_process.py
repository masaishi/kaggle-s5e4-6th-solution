import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from config import cfg
from feature_eng import add_te, feature_eng, preprocess

# from utils import get_index_splits


def calc_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def get_dfs(cfg=cfg):
    # Read CSV files using polars
    df_train = pl.read_csv(cfg.train_path)
    df_train = df_train.filter(pl.col("Number_of_Ads").is_not_null())
    # df_test = pl.read_csv(cfg.test_path)
    df_test = None

    # df_train = get_index_splits(df_train)
    # breakpoint()

    # df_train = df_train.drop("id")
    # df_test = df_test.drop("id")

    target_col = "Listening_Time_minutes"
    y_train = df_train[target_col]
    X_train = df_train.drop(target_col)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,  # 20% for validation
        random_state=42,
    )

    # # Merge with df_podcast
    # df_pltpd = pl.read_csv(cfg.pltpd_path)
    # df_pltpd = df_pltpd.filter(pl.col("Listening_Time_minutes").is_not_null())

    # # Extract target column and prepare for concatenation
    # y_train_pltpd = df_pltpd["Listening_Time_minutes"]
    # df_pltpd = df_pltpd.drop("Listening_Time_minutes")
    # df_pltpd = df_pltpd.with_columns(pl.col("Number_of_Ads").cast(pl.Float64))

    # # Now concatenate with matching schemas
    # X_train = pl.concat([X_train, df_pltpd], how="vertical")
    # y_train = pl.concat([y_train, y_train_pltpd], how="vertical")

    # Preprocess dataframes
    X_train = preprocess(X_train)
    X_valid = preprocess(X_valid, X_train)

    # Create combined df_train for feature engineering
    df_train = X_train.with_columns(y_train.alias("Listening_Time_minutes"))

    # Feature engineering
    X_train = feature_eng(X_train, df_train)
    X_valid = feature_eng(X_valid, df_train)

    # Encode target
    te_dict = add_te(y_train, X_train, X_valid)
    X_train, X_valid, _ = te_dict.values()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
    }
