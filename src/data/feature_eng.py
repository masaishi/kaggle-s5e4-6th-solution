import gc
from itertools import combinations

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.preprocessing import TargetEncoder
from tqdm import tqdm

from config import cfg
from data.data_class import DatasetX
from data.default_selecteds import default_selecteds

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


pl_i_type = pl.Int64
pl_f_type = pl.Float64


def cast_numeric_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    float_cols = [col for col in df.columns if df.schema[col] == pl.Float64 or df.schema[col] == pl.Float32]
    int_cols = [col for col in df.columns if df.schema[col] == pl.Int64 or df.schema[col] == pl.Int32]

    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl_f_type) for col in float_cols])
    if int_cols:
        df = df.with_columns([pl.col(col).cast(pl_i_type) for col in int_cols])

    return df


def preprocess(df: pl.DataFrame, df_train: pl.DataFrame = None) -> pl.DataFrame:
    df = cast_numeric_dtypes(df)
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

    # e_median = df_train.select(pl.col("Episode_Length_minutes").median()).item()
    # g_median = df_train.select(pl.col("Guest_Popularity_percentage").median()).item()
    # n_median = df_train.select(pl.col("Number_of_Ads").median()).item()

    df = df.with_columns(
        pl.col("Episode_Length_minutes").fill_null(-2),
        pl.col("Guest_Popularity_percentage").fill_null(-2),
        pl.col("Number_of_Ads").fill_null(-2),
    )

    return df


def feature_eng(df: pl.DataFrame, df_train: pl.DataFrame) -> pl.DataFrame:
    global selected
    # Cyclical features for day and time
    df = df.with_columns(
        # Day features
        pl.col("Publication_Day").cast(pl_f_type).mul(2 * np.pi / 7).sin().alias("Day_sin"),
        pl.col("Publication_Day").cast(pl_f_type).mul(2 * np.pi / 7).cos().alias("Day_cos"),
        pl.col("Publication_Day").cast(pl_f_type).mul(4 * np.pi / 7).sin().alias("Day_sin2"),
        pl.col("Publication_Day").cast(pl_f_type).mul(4 * np.pi / 7).cos().alias("Day_cos2"),
        # Time features
        pl.col("Publication_Time").cast(pl_f_type).mul(2 * np.pi / 4).sin().alias("Time_sin"),
        pl.col("Publication_Time").cast(pl_f_type).mul(2 * np.pi / 4).cos().alias("Time_cos"),
        pl.col("Publication_Time").cast(pl_f_type).mul(4 * np.pi / 24).sin().alias("Time_sin2"),
        pl.col("Publication_Time").cast(pl_f_type).mul(4 * np.pi / 24).cos().alias("Time_cos2"),
        # Ratio features
        (pl.col("Episode_Length_minutes") / (pl.col("Number_of_Ads") + 1)).fill_null(0).alias("Length_per_Ads"),
        (pl.col("Episode_Length_minutes") / (pl.col("Host_Popularity_percentage") + 1)).fill_null(0).alias("Length_per_Host"),
        (pl.col("Episode_Length_minutes") / (pl.col("Guest_Popularity_percentage") + 1)).fill_null(0).alias("Length_per_Guest"),
        # Episode length features
        pl.col("Episode_Length_minutes").floor().alias("ELen_Int"),
        (pl.col("Episode_Length_minutes") - pl.col("Episode_Length_minutes").floor()).alias("ELen_Dec"),
        pl.col("Host_Popularity_percentage").floor().alias("HPperc_Int"),
        (pl.col("Host_Popularity_percentage") - pl.col("Host_Popularity_percentage").floor()).alias("HPperc_Dec"),
        # Sentiment features
        (pl.col("Episode_Sentiment") == "2").cast(pl.Int8).alias("Is_Positive_Sentiment"),
        pl.when(pl.col("Episode_Sentiment") == "2").then(0.75).otherwise(0.717).cast(pl_f_type).alias("Sentiment_Multiplier"),
        # Squared features
        (pl.col("Episode_Length_minutes") ** 2).alias("Episode_Length_squared"),
        (pl.col("Episode_Length_minutes") ** 3).alias("Episode_Length_squared2"),
    )

    df = df.with_columns(
        (np.sin(2 * np.pi * pl.col("Episode_Num") / 100)).alias("Long_Term_Cycle_Sin"),
        (np.cos(2 * np.pi * pl.col("Episode_Num") / 100)).alias("Long_Term_Cycle_Cos"),
        (pl.col("Episode_Length_minutes") * pl.col("Sentiment_Multiplier")).alias("Expected_Listening_Time_Sentiment"),
    )

    df = df.with_columns(
        (
            (pl.col("Episode_Length_minutes") - pl.col("Episode_Length_minutes").median()).pow(2)
            + (pl.col("Host_Popularity_percentage") - pl.col("Host_Popularity_percentage").median()).pow(2)
            + (pl.col("Guest_Popularity_percentage") - pl.col("Guest_Popularity_percentage").median()).pow(2)
            + (pl.col("Number_of_Ads") - pl.col("Number_of_Ads").median()).pow(2)
        ).alias("Diff_Squared")
    )

    # Convert columns to categorical
    for col in ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Num"]:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))

    return df


def get_combinations(df: pl.DataFrame, columns_to_encode: list, pair_sizes: list) -> list:
    df_length = len(df)

    target_ratios = []
    target_ratios.extend(np.arange(0.01, 0.3, 0.05).tolist())
    target_ratios.extend(np.arange(0.3, 1.01, 0.005).tolist())

    if hasattr(cfg, "eval") and cfg.eval:
        target_ratios = np.arange(0.001, 0.999, 0.05).tolist()

    all_combinations = []
    for r in pair_sizes:
        for cols in combinations(columns_to_encode, r):
            group_counts = len(df.group_by(cols).count())
            ratio = group_counts / df_length
            all_combinations.append((cols, ratio))

    unique_combinations = set()
    for target in target_ratios:
        closest_combination = min(all_combinations, key=lambda x: abs(x[1] - target))
        unique_combinations.add(closest_combination[0])

    return list(unique_combinations)


def cols_encode(df: pl.DataFrame, combinations_list: list, round_num: int = 10) -> pl.DataFrame:
    batch_size = 20
    for i in range(0, len(combinations_list), batch_size):
        batch = combinations_list[i : i + batch_size]

        for cols in tqdm(batch):
            new_col_name = "colen_" + "_".join(cols)
            if cols[0] in df.select(cs.numeric()).columns:
                concat_expr = pl.col(cols[0]).round(round_num).cast(pl.Utf8)
            else:
                concat_expr = pl.col(cols[0]).cast(pl.Utf8)

            for col_name in cols[1:]:
                if col_name in df.select(cs.numeric()).columns:
                    concat_expr = concat_expr + "_" + pl.col(col_name).round(round_num).cast(pl.Utf8)
                else:
                    concat_expr = concat_expr + "_" + pl.col(col_name).cast(pl.Utf8)

            df = df.with_columns(concat_expr.alias(new_col_name).cast(pl.Categorical))

        gc.collect()

        mem_usage = sum(df.estimated_size() for col in df.columns) / (1024 * 1024)
        print(f"Memory usage: {mem_usage:.2f} MB")

    return df


def encode_target(
    target: pl.Series, encode_columns: list, X_train: pl.DataFrame, X_valid: pl.DataFrame, X_test: pl.DataFrame = None, random_state: int = cfg.random_state
) -> DatasetX:
    if isinstance(target, pl.Series):
        target_values = target.to_numpy()
    else:
        target_values = target

    encoder = TargetEncoder(random_state=random_state)

    for col in tqdm(encode_columns, desc="Encoding cols"):
        encoded_col_name = f"{col}_{target.name}_encoded"

        X_train_col = X_train[col].to_numpy().reshape(-1, 1)
        encoded_train = encoder.fit_transform(X_train_col, target_values)
        X_train = X_train.with_columns(pl.Series(encoded_col_name, encoded_train.flatten()))

        X_valid_col = X_valid[col].to_numpy().reshape(-1, 1)
        encoded_valid = encoder.transform(X_valid_col)
        X_valid = X_valid.with_columns(pl.Series(encoded_col_name, encoded_valid.flatten()))

        if X_test is not None:
            X_test_col = X_test[col].to_numpy().reshape(-1, 1)
            encoded_test = encoder.transform(X_test_col)
            X_test = X_test.with_columns(pl.Series(encoded_col_name, encoded_test.flatten()))

        gc.collect()

    return DatasetX(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
    )


def add_te(y_train: pl.Series, X_train: pl.DataFrame, X_valid: pl.DataFrame, X_test: pl.DataFrame = None) -> DatasetX:
    before_encode_len = len(X_train.columns)

    combinations_list = [item.split("-") for item in default_selecteds]

    combinations_list = list(set(tuple(sorted(combo)) for combo in combinations_list))

    print("Combinations list length:", len(combinations_list))

    X_train = cols_encode(X_train, combinations_list)
    X_valid = cols_encode(X_valid, combinations_list)
    if X_test is not None:
        X_test = cols_encode(X_test, combinations_list)

    encoded_columns = X_train.columns[before_encode_len:]

    datasetX = encode_target(y_train, encoded_columns, X_train, X_valid, X_test=X_test)
    X_train, X_valid, X_test = datasetX.get()

    # encoded_columns = [col for col in encoded_columns if "Episode_Length_minutes" not in col and "ELen" not in col and "Length_per" not in col]
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

    pte_selecteds = [
        "Podcast_Name-Episode_Num-Length_per_Guest-HPperc_Dec",
        "Podcast_Name-Host_Popularity_percentage-Episode_Num-Length_per_Guest",
        "Genre-Publication_Day-Length_per_Guest-HPperc_Dec",
        "Podcast_Name-Host_Popularity_percentage-Episode_Sentiment-Length_per_Guest",
        "Genre-Host_Popularity_percentage-Episode_Sentiment-Length_per_Guest",
        "Podcast_Name-Episode_Sentiment-Length_per_Guest-HPperc_Dec",
        "Podcast_Name-Publication_Day-Length_per_Guest-HPperc_Int",
        "Host_Popularity_percentage-Publication_Day-Episode_Sentiment-Length_per_Guest",
        "Podcast_Name-Guest_Popularity_percentage-Length_per_Ads-HPperc_Int",
        "Podcast_Name-Publication_Time-Length_per_Guest-HPperc_Int",
        "Host_Popularity_percentage-Publication_Day-Length_per_Guest",
        "Podcast_Name-Episode_Sentiment-Length_per_Guest-HPperc_Int",
        "Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-ELen_Dec",
        "Host_Popularity_percentage-Publication_Time-Length_per_Guest",
        "Host_Popularity_percentage-Number_of_Ads-Length_per_Guest",
        "Podcast_Name-Publication_Day-Publication_Time-Length_per_Guest",
        "Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-ELen_Dec",
        "Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Sentiment-ELen_Dec",
        "Host_Popularity_percentage-Guest_Popularity_percentage-Episode_Num-ELen_Int",
        "Podcast_Name-Publication_Time-Guest_Popularity_percentage-Length_per_Ads",
        "Podcast_Name-Publication_Time-Number_of_Ads-Length_per_Guest",
        "Podcast_Name-Publication_Day-Number_of_Ads-Length_per_Host",
        "Guest_Popularity_percentage-Episode_Num-ELen_Int-HPperc_Dec",
        "Podcast_Name-Publication_Time-Episode_Sentiment-Length_per_Guest",
        "Host_Popularity_percentage-Guest_Popularity_percentage-ELen_Dec",
        "Episode_Length_minutes-Guest_Popularity_percentage-HPperc_Dec",
        "Length_per_Guest-ELen_Dec-HPperc_Dec",
        "Publication_Day-Publication_Time-Episode_Num-Length_per_Host",
        "Publication_Day-Number_of_Ads-Episode_Num-Length_per_Host",
        "Podcast_Name-Publication_Day-Length_per_Host",
        "Podcast_Name-Episode_Length_minutes-Guest_Popularity_percentage-Episode_Sentiment",
        "Host_Popularity_percentage-Publication_Day-Guest_Popularity_percentage-ELen_Int",
        "Publication_Time-Episode_Sentiment-Episode_Num-Length_per_Host",
        "Publication_Time-Episode_Sentiment-Length_per_Guest-HPperc_Int",
        "Publication_Day-Episode_Num-Length_per_Host-ELen_Int",
        "Publication_Day-Episode_Num-Length_per_Host",
        "Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Episode_Num",
        "Episode_Length_minutes-Publication_Time-Guest_Popularity_percentage-Episode_Num",
        "Publication_Time-Episode_Num-Length_per_Guest-ELen_Int",
        "Publication_Time-Episode_Num-Length_per_Guest",
        "Genre-Publication_Day-Length_per_Host-HPperc_Int",
        "Genre-Publication_Day-Length_per_Host-ELen_Int",
        "Episode_Length_minutes-Genre-Host_Popularity_percentage-Publication_Day",
        "Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-ELen_Int",
        "Podcast_Name-Guest_Popularity_percentage-Episode_Num-ELen_Int",
        "Episode_Length_minutes-Host_Popularity_percentage-Publication_Time-Episode_Num",
        "Episode_Sentiment-Episode_Num-Length_per_Ads-HPperc_Dec",
        "Podcast_Name-Host_Popularity_percentage-Episode_Num-ELen_Int",
        "Publication_Day-Publication_Time-Number_of_Ads-Length_per_Guest",
        "Publication_Day-Episode_Sentiment-Length_per_Ads-Length_per_Guest",
        "Host_Popularity_percentage-Guest_Popularity_percentage-ELen_Int",
        "Podcast_Name-Episode_Length_minutes-Episode_Num-HPperc_Int",
        "Genre-Guest_Popularity_percentage-Episode_Num-ELen_Int",
        "Guest_Popularity_percentage-Episode_Num-ELen_Int-HPperc_Int",
        "Publication_Day-Number_of_Ads-Length_per_Guest-ELen_Int",
        "Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage-Number_of_Ads",
        "Publication_Day-Guest_Popularity_percentage-Length_per_Ads-ELen_Dec",
        "Episode_Length_minutes-Host_Popularity_percentage-Publication_Day-Episode_Sentiment",
        "Publication_Day-Guest_Popularity_percentage-Episode_Num-ELen_Int",
        "Publication_Day-Episode_Num-Length_per_Ads-HPperc_Int",
        "Number_of_Ads-Episode_Sentiment-Length_per_Guest-ELen_Int",
        "Number_of_Ads-Episode_Sentiment-Length_per_Guest-ELen_Dec",
        "Publication_Day-Guest_Popularity_percentage-ELen_Int-HPperc_Int",
        "Host_Popularity_percentage-Publication_Day-Episode_Num-ELen_Int",
        "Publication_Time-Episode_Num-Length_per_Ads-HPperc_Int",
        "Guest_Popularity_percentage-Number_of_Ads-Episode_Num-ELen_Int",
        "Publication_Day-Length_per_Host-ELen_Dec-HPperc_Dec",
        "Episode_Length_minutes-Host_Popularity_percentage-Publication_Day",
        "Publication_Day-Length_per_Host-HPperc_Dec",
        "Guest_Popularity_percentage-Episode_Sentiment-Episode_Num-ELen_Int",
        "Episode_Sentiment-Episode_Num-Length_per_Ads-HPperc_Int",
        "Host_Popularity_percentage-Number_of_Ads-Episode_Num-ELen_Int",
        "Host_Popularity_percentage-Publication_Time-Episode_Num-ELen_Int",
        "Guest_Popularity_percentage-Number_of_Ads-ELen_Int-HPperc_Int",
        "Publication_Time-Guest_Popularity_percentage-ELen_Int-HPperc_Int",
        "Host_Popularity_percentage-Episode_Sentiment-Episode_Num-ELen_Int",
        "Guest_Popularity_percentage-Episode_Sentiment-ELen_Int-HPperc_Int",
        "Episode_Length_minutes-Publication_Time-Episode_Num-HPperc_Int",
        "Episode_Length_minutes-Episode_Sentiment-Episode_Num-HPperc_Int",
        "Guest_Popularity_percentage-Episode_Num-ELen_Int",
        "Episode_Length_minutes-Genre-Publication_Day-HPperc_Int",
        "Host_Popularity_percentage-Episode_Num-ELen_Int",
        "Guest_Popularity_percentage-ELen_Int-HPperc_Int",
        "Genre-Host_Popularity_percentage-Publication_Time-ELen_Int",
        "Podcast_Name-Episode_Num-ELen_Int-HPperc_Int",
        "Episode_Length_minutes-Episode_Num-HPperc_Int",
        "Episode_Length_minutes-Host_Popularity_percentage",
        "Length_per_Host-ELen_Int",
        "Episode_Length_minutes-Genre-Publication_Time-HPperc_Int",
        "Publication_Day-Guest_Popularity_percentage-Number_of_Ads-ELen_Int",
        "Publication_Day-Publication_Time-Guest_Popularity_percentage-ELen_Int",
        "Publication_Day-Guest_Popularity_percentage-Episode_Sentiment-ELen_Int",
        "Episode_Length_minutes-Publication_Day-Publication_Time-Episode_Num",
        "Host_Popularity_percentage-Publication_Day-Number_of_Ads-ELen_Int",
        "Host_Popularity_percentage-Publication_Day-Publication_Time-ELen_Int",
        "Episode_Length_minutes-Publication_Day-Publication_Time-HPperc_Int",
        "Host_Popularity_percentage-Publication_Day-Episode_Sentiment-ELen_Int",
        "Podcast_Name-Episode_Length_minutes-Publication_Day-Publication_Time",
        "Publication_Day-Length_per_Ads-ELen_Dec-HPperc_Int",
        "Host_Popularity_percentage-Publication_Time-Number_of_Ads-ELen_Int",
        "Episode_Length_minutes-Publication_Day-Episode_Sentiment-HPperc_Int",
        "Episode_Length_minutes-Publication_Time-Episode_Sentiment-Episode_Num",
        "Host_Popularity_percentage-Publication_Time-Episode_Sentiment-ELen_Int",
        "Episode_Length_minutes-Publication_Time-Number_of_Ads-HPperc_Int",
        "Publication_Time-Length_per_Ads-ELen_Dec-HPperc_Int",
        "Publication_Day-Episode_Num-ELen_Int-HPperc_Int",
        "Episode_Length_minutes-Number_of_Ads-Episode_Sentiment-HPperc_Int",
        "Episode_Length_minutes-Publication_Day-Episode_Num",
        "Host_Popularity_percentage-Publication_Day-ELen_Int",
        "Episode_Length_minutes-Publication_Day-HPperc_Int",
        "Podcast_Name-Publication_Day-Episode_Num-ELen_Int",
        "Publication_Time-Episode_Num-ELen_Int-HPperc_Int",
        "Number_of_Ads-Episode_Num-ELen_Int-HPperc_Int",
        "Host_Popularity_percentage-Publication_Time-ELen_Int",
        "Episode_Sentiment-Episode_Num-ELen_Int-HPperc_Int",
        "Host_Popularity_percentage-Episode_Sentiment-ELen_Int",
        "Podcast_Name-Publication_Time-Episode_Num-ELen_Int",
        "Podcast_Name-Episode_Length_minutes-Number_of_Ads",
        "Podcast_Name-Episode_Length_minutes-Genre-Number_of_Ads",
        "Episode_Length_minutes-Genre-Publication_Time-Number_of_Ads",
        "Podcast_Name-Episode_Sentiment-ELen_Int-HPperc_Int",
        "Episode_Num-ELen_Int-HPperc_Int",
        "Episode_Length_minutes-Publication_Day-Publication_Time-Number_of_Ads",
        "Episode_Length_minutes-HPperc_Int",
        "Episode_Length_minutes-Publication_Day-Number_of_Ads-Episode_Sentiment",
        "Episode_Length_minutes-Publication_Time-Number_of_Ads",
        "Episode_Length_minutes-Publication_Time-Number_of_Ads-Episode_Sentiment",
        "Publication_Time-Episode_Sentiment-Length_per_Ads-ELen_Dec",
        "Publication_Time-Number_of_Ads-ELen_Int-HPperc_Int",
        "Episode_Length_minutes-Genre-Number_of_Ads",
        "Genre-Length_per_Ads-ELen_Dec",
        "Genre-Number_of_Ads-ELen_Int-HPperc_Int",
        "Publication_Day-Number_of_Ads-ELen_Int-HPperc_Int",
    ]

    # selecteds = random.sample(before_fe_selecteds, min(len(before_fe_selecteds), 50))
    combinations_list = []
    combinations_list += [item.split("-") for item in default_selecteds]
    # combinations_list += [item.split("-") for item in pte_selecteds]

    m = df_pltpd["Listening_Time_minutes"].mean()

    for cols in combinations_list:
        n = f"pte_{'_'.join(cols)}"
        means = df_pltpd.group_by(cols).agg(pl.col("Listening_Time_minutes").mean().alias("mean_listening_time"))
        df = df.join(means, on=cols, how="left").with_columns(pl.col("mean_listening_time").fill_null(m).alias(n)).drop("mean_listening_time")

    return df
