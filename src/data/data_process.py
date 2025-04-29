import polars as pl

from config import cfg
from data.data_class import DatasetXy, Dfs
from data.feature_eng import add_original_cols, add_te, feature_eng, preprocess
from data.simple_feature_eng import standardize

_ = [DatasetXy, Dfs, add_te, feature_eng, preprocess, add_original_cols, standardize]


def get_dfs(cfg=cfg) -> Dfs:
    df_train = pl.read_csv(cfg.train_path)
    df_train = df_train.filter(pl.col("Number_of_Ads").is_not_null())

    df_test = None
    if hasattr(cfg, "predict") and cfg.predict:
        df_test = pl.read_csv(cfg.test_path)

    return Dfs(df_train=df_train, df_test=df_test)


def add_fold(df: pl.DataFrame) -> pl.DataFrame:
    cols = ["Podcast_Name", "Episode_Title", "Host_Popularity_percentage", "Publication_Day"]
    concat_expr = pl.col(cols[0]).cast(pl.Utf8)
    for col_name in cols[1:]:
        concat_expr = concat_expr + "_" + pl.col(col_name).cast(pl.Utf8)

    df = df.with_columns(concat_expr.alias("fold").cast(pl.Categorical))
    return df


def get_Xy(dfs: Dfs) -> DatasetXy:
    df_train, df_valid, df_test = dfs.get()

    df_train = preprocess(df_train)
    df_valid = preprocess(df_valid, df_train)
    if df_test is not None:
        df_test = preprocess(df_test, df_train)

    target_col = "Listening_Time_minutes"
    y_train = df_train[target_col]
    X_train = df_train.drop(target_col)
    y_valid = df_valid[target_col]
    X_valid = df_valid.drop(target_col)
    X_test = df_test

    X_train = feature_eng(X_train, df_train)
    X_valid = feature_eng(X_valid, df_train)
    if X_test is not None:
        X_test = feature_eng(X_test, df_train)

    df_pltpd = pl.read_csv(cfg.pltpd_path)
    df_pltpd = df_pltpd.drop_nulls(subset=["Listening_Time_minutes"])
    df_pltpd = df_pltpd.filter(pl.col("Episode_Length_minutes").is_not_null())
    df_pltpd = df_pltpd.with_columns(
        pl.col("Number_of_Ads").cast(pl.Float64),
    )
    df_pltpd = add_fold(df_pltpd)
    df_pltpd = preprocess(df_pltpd)
    df_pltpd = feature_eng(df_pltpd, df_train)
    df_pltpd = df_pltpd.with_columns(
        (pl.lit(1000000).cast(pl.Int64) + pl.arange(0, len(df_pltpd))).cast(pl.Int64).alias("id"),
    )

    y_train = pl.concat([y_train, df_pltpd["Listening_Time_minutes"]], how="vertical")
    X_train = pl.concat([X_train, df_pltpd.select(X_train.columns)], how="vertical")

    X_train = add_original_cols(X_train, df_pltpd)
    X_valid = add_original_cols(X_valid, df_pltpd)
    if X_test is not None:
        X_test = add_original_cols(X_test, df_pltpd)

    datasetX = add_te(y_train, X_train, X_valid, X_test)
    X_train, X_valid, X_test = datasetX.get()

    X_train = standardize(X_train, df_train)
    X_valid = standardize(X_valid, df_train)
    if X_test is not None:
        X_test = standardize(X_test, df_train)

    X_train = X_train.drop(["id", "fold"])
    X_valid = X_valid.drop(["id", "fold"])
    if X_test is not None:
        X_test = X_test.drop(["id", "fold"])

    return DatasetXy(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=None)
