import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from config import cfg
from data.data_class import DatasetX
from data.feature_eng import cols_encode, encode_target

selecteds = [
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage",
    "Episode_Length_minutes-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Episode_Length_minutes-Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment",
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage",
    "Episode_Length_minutes-Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Num-Episode_Length_minutes-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Podcast_Name-Episode_Num-Episode_Length_minutes-Guest_Popularity_percentage",
    "Episode_Length_minutes-Genre-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Episode_Num-Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage",
    "Podcast_Name-Episode_Length_minutes-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Podcast_Name-Episode_Num-Episode_Length_minutes-Host_Popularity_percentage",
    "Episode_Num-Episode_Length_minutes-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Num-Episode_Length_minutes-Publication_Time-Guest_Popularity_percentage",
    "Podcast_Name-Episode_Length_minutes-Publication_Time-Guest_Popularity_percentage",
    "Episode_Num-Episode_Length_minutes-Guest_Popularity_percentage-Episode_Sentiment",
    "Episode_Num-Episode_Length_minutes-Host_Popularity_percentage-Publication_Day",
    "Episode_Num-Episode_Length_minutes-Genre-Guest_Popularity_percentage",
    "Episode_Num-Episode_Length_minutes-Genre-Host_Popularity_percentage",
    "Podcast_Name-Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage",
    "Episode_Num-Episode_Length_minutes-Host_Popularity_percentage-Publication_Time",
    "Podcast_Name-Episode_Length_minutes-Guest_Popularity_percentage-Number_of_Ads",
    "Podcast_Name-Episode_Length_minutes-Host_Popularity_percentage-Publication_Day",
    "Episode_Num-Episode_Length_minutes-Guest_Popularity_percentage",
    "Episode_Num-Episode_Length_minutes-Host_Popularity_percentage-Number_of_Ads",
    "Podcast_Name-Episode_Length_minutes-Guest_Popularity_percentage-Episode_Sentiment",
    "Episode_Num-Episode_Length_minutes-Host_Popularity_percentage-Episode_Sentiment",
    "Podcast_Name-Episode_Length_minutes-Host_Popularity_percentage-Publication_Time",
    "Podcast_Name-Episode_Num-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Podcast_Name-Episode_Length_minutes-Host_Popularity_percentage-Number_of_Ads",
    "Podcast_Name-Episode_Length_minutes-Host_Popularity_percentage-Episode_Sentiment",
    "Episode_Num-Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage",
    "Episode_Num-Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment",
    "Episode_Length_minutes-Genre-Publication_Day-Guest_Popularity_percentage",
    "Episode_Num-Episode_Length_minutes-Host_Popularity_percentage",
    "Podcast_Name-Episode_Length_minutes-Guest_Popularity_percentage",
    "Podcast_Name-Episode_Length_minutes-Genre-Guest_Popularity_percentage",
    "Episode_Length_minutes-Genre-Publication_Time-Guest_Popularity_percentage",
    "Episode_Num-Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage",
    "Episode_Num-Genre-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Episode_Length_minutes-Genre-Host_Popularity_percentage-Publication_Day",
    "Episode_Length_minutes-Publication_Day-Publication_Time-Guest_Popularity_percentage",
    "Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Length_minutes-Genre-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Num-Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage-Episode_Sentiment",
    "Podcast_Name-Episode_Length_minutes-Host_Popularity_percentage",
    "Podcast_Name-Episode_Length_minutes-Genre-Host_Popularity_percentage",
    "Episode_Length_minutes-Genre-Host_Popularity_percentage-Publication_Time",
    "Podcast_Name-Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage",
    "Episode_Length_minutes-Genre-Guest_Popularity_percentage-Episode_Sentiment",
    "Podcast_Name-Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Length_minutes-Publication_Time-Guest_Popularity_percentage-Number_of_Ads",
    "Podcast_Name-Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage",
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Publication_Time",
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Number_of_Ads",
    "Episode_Length_minutes-Genre-Host_Popularity_percentage-Number_of_Ads",
    "Podcast_Name-Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment",
    "Episode_Length_minutes-Publication_Time-Guest_Popularity_percentage-Episode_Sentiment",
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Episode_Sentiment",
    "Episode_Num-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Episode_Length_minutes-Guest_Popularity_percentage-Number_of_Ads-Episode_Sentiment",
    "Episode_Length_minutes-Genre-Host_Popularity_percentage-Episode_Sentiment",
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Time-Number_of_Ads",
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Time-Episode_Sentiment",
    "Episode_Length_minutes-Genre-Guest_Popularity_percentage",
    "Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage",
    "Episode_Length_minutes-Host_Popularity_percentage-Number_of_Ads-Episode_Sentiment",
    "Genre-Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage",
    "Podcast_Name-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Podcast_Name-Genre-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Host_Popularity_percentage-Publication_Day-Publication_Time-Guest_Popularity_percentage",
    "Genre-Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage",
    "Episode_Length_minutes-Genre-Host_Popularity_percentage",
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Day",
    "Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Length_minutes-Publication_Time-Guest_Popularity_percentage",
    "Genre-Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Length_minutes-Guest_Popularity_percentage-Number_of_Ads",
    "Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-Episode_Sentiment",
    "Genre-Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment",
    "Episode_Length_minutes-Guest_Popularity_percentage-Episode_Sentiment",
    "Podcast_Name-Episode_Num-Episode_Length_minutes-Publication_Day",
    "Episode_Length_minutes-Host_Popularity_percentage-Publication_Time",
    "Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Length_minutes-Host_Popularity_percentage-Number_of_Ads",
    "Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-Episode_Sentiment",
    "Episode_Length_minutes-Host_Popularity_percentage-Episode_Sentiment",
    "Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads-Episode_Sentiment",
    "Podcast_Name-Episode_Num-Episode_Length_minutes-Publication_Time",
    "Podcast_Name-Episode_Num-Episode_Length_minutes-Number_of_Ads",
    "Podcast_Name-Episode_Num-Episode_Length_minutes-Episode_Sentiment",
    "Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage",
    "Episode_Length_minutes-Guest_Popularity_percentage",
    "Genre-Host_Popularity_percentage-Guest_Popularity_percentage",
    "Podcast_Name-Episode_Num-Publication_Day-Guest_Popularity_percentage",
    "Episode_Length_minutes-Host_Popularity_percentage",
    "Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage",
    "Host_Popularity_percentage-Guest_Popularity_percentage-Number_of_Ads",
    "Episode_Num-Episode_Length_minutes-Genre-Publication_Day",
    "Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment",
]
if hasattr(cfg, "eval") and cfg.eval:
    selecteds = [selecteds[i] for i in range(0, len(selecteds), 5)]


def calc_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


re_dict = {}
re_dict["podc_dict"] = {
    "Mystery Matters": 0,
    "Joke Junction": 1,
    "Study Sessions": 2,
    "Digital Digest": 3,
    "Mind & Body": 4,
    "Fitness First": 5,
    "Criminal Minds": 6,
    "News Roundup": 7,
    "Daily Digest": 8,
    "Music Matters": 9,
    "Sports Central": 10,
    "Melody Mix": 11,
    "Game Day": 12,
    "Gadget Geek": 13,
    "Global News": 14,
    "Tech Talks": 15,
    "Sport Spot": 16,
    "Funny Folks": 17,
    "Sports Weekly": 18,
    "Business Briefs": 19,
    "Tech Trends": 20,
    "Innovators": 21,
    "Health Hour": 22,
    "Comedy Corner": 23,
    "Sound Waves": 24,
    "Brain Boost": 25,
    "Athlete's Arena": 26,
    "Wellness Wave": 27,
    "Style Guide": 28,
    "World Watch": 29,
    "Humor Hub": 30,
    "Money Matters": 31,
    "Healthy Living": 32,
    "Home & Living": 33,
    "Educational Nuggets": 34,
    "Market Masters": 35,
    "Learning Lab": 36,
    "Lifestyle Lounge": 37,
    "Crime Chronicles": 38,
    "Detective Diaries": 39,
    "Life Lessons": 40,
    "Current Affairs": 41,
    "Finance Focus": 42,
    "Laugh Line": 43,
    "True Crime Stories": 44,
    "Business Insights": 45,
    "Fashion Forward": 46,
    "Tune Time": 47,
}
re_dict["genr_dict"] = {
    "True Crime": 0,
    "Comedy": 1,
    "Education": 2,
    "Technology": 3,
    "Health": 4,
    "News": 5,
    "Music": 6,
    "Sports": 7,
    "Business": 8,
    "Lifestyle": 9,
}
re_dict["week_dict"] = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}
re_dict["time_dict"] = {"Morning": 10, "Afternoon": 14, "Evening": 17, "Night": 21}
re_dict["sent_dict"] = {"Negative": 0, "Neutral": 1, "Positive": 2}

pl_i_type = pl.Int32
pl_f_type = pl.Float32

GROUP_SPLIT = 50


def cast_numeric_dtypes(df):
    float_cols = [col for col in df.columns if df.schema[col] == pl.Float64 or df.schema[col] == pl.Float32]
    int_cols = [col for col in df.columns if df.schema[col] == pl.Int64 or df.schema[col] == pl.Int32]

    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl_f_type) for col in float_cols])
    if int_cols:
        df = df.with_columns([pl.col(col).cast(pl_i_type) for col in int_cols])

    return df


def preprocess(df, df_train=None):
    # df = cast_numeric_dtypes(df)
    df = df.with_columns(pl.col("Episode_Title").str.slice(8).cast(pl.Int32).alias("Episode_Num")).drop("Episode_Title")

    # Convert categorical variables using mapping
    for col, mapping in [
        ("Genre", re_dict["genr_dict"]),
        ("Podcast_Name", re_dict["podc_dict"]),
        ("Publication_Day", re_dict["week_dict"]),
        ("Publication_Time", re_dict["time_dict"]),
        ("Episode_Sentiment", re_dict["sent_dict"]),
    ]:
        df = df.with_columns(pl.col(col).replace(mapping).alias(col))

    # Cap extreme values
    df = df.with_columns(
        pl.when(pl.col("Episode_Length_minutes") > 121.0).then(121.0).otherwise(pl.col("Episode_Length_minutes")).alias("Episode_Length_minutes"),
        pl.when(pl.col("Number_of_Ads") > 103.91).then(103.91).otherwise(pl.col("Number_of_Ads")).alias("Number_of_Ads"),
    )

    # Create NaN indicator columns
    df = df.with_columns(
        pl.col("Episode_Length_minutes").is_null().cast(pl.Utf8).cast(pl.Categorical).alias("Episode_Length_minutes_NaN"),
        pl.col("Guest_Popularity_percentage").is_null().cast(pl.Utf8).cast(pl.Categorical).alias("Guest_Popularity_percentage_NaN"),
    )

    # Fill NA values with median
    if df_train is None:
        df_train = df.clone()

    e_median = df_train.select(pl.col("Episode_Length_minutes").median()).item()
    g_median = df_train.select(pl.col("Guest_Popularity_percentage").median()).item()
    n_median = df_train.select(pl.col("Number_of_Ads").median()).item()

    df = df.with_columns(
        pl.col("Episode_Length_minutes").fill_null(e_median),
        pl.col("Guest_Popularity_percentage").fill_null(g_median),
        pl.col("Number_of_Ads").fill_null(n_median),
    )

    df = df.with_columns(
        pl.col("Episode_Num").cast(pl.Utf8).cast(pl.Categorical).alias("Episode_Num_Cat"),
    )

    return df


def feature_eng(df, df_train, n_splits=GROUP_SPLIT):
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
            smoothing = np.random.randint(5, 15)
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
    df = df.drop(categorical_cols)

    return df


def add_te(y_train: pl.Series, X_train: pl.DataFrame, X_valid: pl.DataFrame, X_test: pl.DataFrame = None) -> DatasetX:
    before_encode_len = len(X_train.columns)

    combinations_list = [item.split("-") for item in selecteds]

    print("Combinations list length:", len(combinations_list))
    print("Combinations list:", combinations_list)

    X_train = cols_encode(X_train, combinations_list)
    X_valid = cols_encode(X_valid, combinations_list)
    if X_test is not None:
        X_test = cols_encode(X_test, combinations_list)

    encoded_columns = X_train.columns[before_encode_len:]
    print("Length of train columns:", before_encode_len)

    datasetX = encode_target(y_train, encoded_columns, X_train, X_valid, X_test=X_test)
    X_train, X_valid, X_test = datasetX.get()

    # encoded_columns = [col for col in encoded_columns if col != "Episode_Length_minutes"]
    # datasetX = encode_target(X_train["Episode_Length_minutes"], encoded_columns, X_train, X_valid, X_test=X_test)
    # X_train, X_valid, X_test = datasetX.get()

    X_train = X_train.drop(encoded_columns)
    X_valid = X_valid.drop(encoded_columns)
    if X_test is not None:
        X_test = X_test.drop(encoded_columns)

    return DatasetX(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
    )


def add_original_cols(df: pl.DataFrame, df_pltpd: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = df.select(cs.numeric()).columns
    if "id" in numeric_cols:
        numeric_cols.remove("id")

    # combinations_list = [[col] for col in numeric_cols]

    # combinations_list = [item.split("-") for item in selecteds[:20]]
    combinations_list = [item.split("-") for item in selecteds]
    combinations_list += [[col] for col in numeric_cols]

    m = df_pltpd["Listening_Time_minutes"].mean()

    for cols in combinations_list:
        n = f"pte-{'_'.join(cols)}"
        means = df_pltpd.group_by(cols).agg(pl.col("Listening_Time_minutes").mean().alias("mean_listening_time"))
        df = df.join(means, on=cols, how="left").with_columns(pl.col("mean_listening_time").fill_null(m).alias(n)).drop("mean_listening_time")

    return df
