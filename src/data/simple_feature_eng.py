import polars as pl
from sklearn.model_selection import GroupKFold

GROUP_SPLIT = 2


def standardize(df: pl.DataFrame, df_train: pl.DataFrame, n_splits: int = GROUP_SPLIT) -> pl.DataFrame:
    for col in ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Num"]:
        df_train = df_train.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))
    df_train = df_train.with_columns(
        pl.col("Episode_Num").cast(pl.Utf8).cast(pl.Categorical).alias("Episode_Num_Cat"),
    )
    df = df.with_columns(
        pl.col("Episode_Num").cast(pl.Utf8).cast(pl.Categorical).alias("Episode_Num_Cat"),
    )

    # numeric_cols = df_train.select(cs.numeric()).columns
    numeric_cols = ["Listening_Time_minutes", "Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"]
    if "id" in numeric_cols:
        numeric_cols.remove("id")

    categorical_cols = [
        "Podcast_Name",
        "Genre",
        "Publication_Day",
        "Publication_Time",
        "Episode_Num_Cat",
        "Episode_Sentiment",
        "Episode_Length_minutes_NaN",
        "Guest_Popularity_percentage_NaN",
    ]

    group_kfold = GroupKFold(n_splits=n_splits)
    df_update = pl.DataFrame()
    for (_, idx_valid), (t_idx_train, _) in zip(group_kfold.split(df, groups=df["fold"]), group_kfold.split(df_train, groups=df_train["fold"])):
        df_train_part = df_train[t_idx_train]
        stats = {
            col: {
                "mean": df_train_part.select(pl.col(col).mean()).item(),
                "std": df_train_part.select(pl.col(col).std()).item(),
            }
            for col in numeric_cols
        }

        # transformations = []
        # transform_cols = [col for col in numeric_cols if col != "Listening_Time_minutes"]
        # for col in transform_cols:
        #     transformations.append(((pl.col(col) - stats[col]["mean"]) / stats[col]["std"]).alias(f"{col}"))
        #     # transformations.append((pl.col(col) - stats[col]["mean"]).alias(f"{col}"))
        # df_update_part = df[idx_valid].with_columns(transformations)

        df_update_part = df[idx_valid]
        df_update_part = df_update_part.with_columns(
            pl.lit(stats["Listening_Time_minutes"]["mean"]).alias("Listening_Time_minutes_mean"),
            # pl.lit(stats["Listening_Time_minutes"]["std"]).alias("Listening_Time_minutes_std"),
        )

        for col in categorical_cols:
            # mean_target = df_train_part.group_by(col).agg(pl.col("Listening_Time_minutes").mean().alias(f"{col}_mean"))

            # df_update_part = df_update_part.join(mean_target, on=col, how="left").with_columns(
            #     pl.col(f"{col}_mean").fill_null(stats["Listening_Time_minutes"]["mean"]).alias(f"{col}_mean")
            # )

            # smoothing = np.random.randint(0, 5)
            smoothing = 0
            target_stats = df_train_part.group_by(col).agg(
                pl.col("Listening_Time_minutes").mean().alias("mean"), pl.col("Listening_Time_minutes").count().alias("count")
            )

            global_mean = stats["Listening_Time_minutes"]["mean"]
            target_stats = target_stats.with_columns(
                ((pl.col("count") * pl.col("mean") + smoothing * global_mean) / (pl.col("count") + smoothing)).alias(f"{col}_mean")
            )
            target_stats = target_stats.select([col, f"{col}_mean"])
            df_update_part = df_update_part.join(target_stats, on=col, how="left").with_columns(
                pl.col(f"{col}_mean").fill_null(stats["Listening_Time_minutes"]["mean"]).alias(f"{col}_mean")
            )

        df_update = pl.concat([df_update, df_update_part], how="vertical")

    df_update = df_update.sort("id")
    df = df.with_columns(df_update)
    # df = df.drop(categorical_cols)

    return df
