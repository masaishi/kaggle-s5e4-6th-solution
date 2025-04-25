import gc
from itertools import combinations

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import cfg


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


def cast_numeric_dtypes(df):
    float_cols = [col for col in df.columns if df.schema[col] == pl.Float64 or df.schema[col] == pl.Float32]
    int_cols = [col for col in df.columns if df.schema[col] == pl.Int64 or df.schema[col] == pl.Int32]

    if float_cols:
        df = df.with_columns([pl.col(col).cast(pl_f_type) for col in float_cols])
    if int_cols:
        df = df.with_columns([pl.col(col).cast(pl_i_type) for col in int_cols])

    return df


def preprocess(df, df_train=None):
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

    e_median = df_train.select(pl.col("Episode_Length_minutes").median()).item()
    g_median = df_train.select(pl.col("Guest_Popularity_percentage").median()).item()
    n_median = df_train.select(pl.col("Number_of_Ads").median()).item()

    df = df.with_columns(
        pl.col("Episode_Length_minutes").fill_null(e_median),
        pl.col("Guest_Popularity_percentage").fill_null(g_median),
        pl.col("Number_of_Ads").fill_null(n_median),
    )

    return df


def feature_eng(df, df_train):
    numeric_cols = df.select(cs.numeric()).columns
    stats = {
        col: {
            "mean": df_train.select(pl.col(col).mean()).item(),
            "std": df_train.select(pl.col(col).std()).item(),
            "min": df_train.select(pl.col(col).min()).item(),
            "max": df_train.select(pl.col(col).max()).item(),
            "median": df_train.select(pl.col(col).median()).item(),
            "q1": df_train.select(pl.col(col).quantile(0.25)).item(),
            "q3": df_train.select(pl.col(col).quantile(0.75)).item(),
        }
        for col in numeric_cols
    }

    transformations = []
    for col in numeric_cols:
        # Existing transformations
        transformations.append(((pl.col(col) - stats[col]["mean"]) / stats[col]["std"]).alias(f"{col}_norm"))
        transformations.append(pl.col(col).pow(2).alias(f"{col}_squared"))
        transformations.append((pl.col(col) + 1).log().alias(f"{col}_log"))
        transformations.append(pl.when(pl.col(col) >= 0).then(pl.col(col).sqrt()).otherwise(0).alias(f"{col}_sqrt"))
        transformations.append(((pl.col(col) - stats[col]["min"]) / (stats[col]["max"] - stats[col]["min"])).alias(f"{col}_minmax"))

        # New transformations
        # Robust scaling using IQR
        transformations.append(((pl.col(col) - stats[col]["median"]) / (stats[col]["q3"] - stats[col]["q1"])).alias(f"{col}_robust"))

        # Binning features (quantile-based)
        transformations.append(
            pl.when(pl.col(col) <= stats[col]["q1"])
            .then(0)
            .when(pl.col(col) <= stats[col]["median"])
            .then(1)
            .when(pl.col(col) <= stats[col]["q3"])
            .then(2)
            .otherwise(3)
            .alias(f"{col}_bin")
        )

        # Cubic transformation
        transformations.append(pl.col(col).pow(3).alias(f"{col}_cubed"))

        # Reciprocal (with safety check)
        transformations.append(pl.when(pl.col(col).abs() > 1e-10).then(1 / pl.col(col)).otherwise(0).alias(f"{col}_recip"))

        # Exponential
        transformations.append(pl.col(col).exp().clip(0, 1e10).alias(f"{col}_exp"))

        # Hyperbolic functions
        transformations.append(pl.col(col).tanh().alias(f"{col}_tanh"))
        transformations.append(pl.col(col).clip(-20, 20).sinh().alias(f"{col}_sinh"))

        # Difference from mean and median
        transformations.append((pl.col(col) - stats[col]["mean"]).alias(f"{col}_diff_mean"))
        transformations.append((pl.col(col) - stats[col]["median"]).alias(f"{col}_diff_median"))

        # Z-score capped (for outlier handling)
        transformations.append(((pl.col(col) - stats[col]["mean"]) / stats[col]["std"]).clip(-3, 3).alias(f"{col}_norm_clipped"))

    df = df.with_columns(transformations)

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
        # Annual patterns (simulated with episode numbers)
        (np.sin(2 * np.pi * pl.col("Episode_Num") / 52)).alias("Annual_Cycle_Sin"),
        (np.cos(2 * np.pi * pl.col("Episode_Num") / 52)).alias("Annual_Cycle_Cos"),
        (np.sin(2 * np.pi * pl.col("Episode_Num") / 13)).alias("Quarterly_Cycle_Sin"),
        (np.cos(2 * np.pi * pl.col("Episode_Num") / 13)).alias("Quarterly_Cycle_Cos"),
        (np.sin(2 * np.pi * pl.col("Episode_Num") / 100)).alias("Long_Term_Cycle_Sin"),
        (np.cos(2 * np.pi * pl.col("Episode_Num") / 100)).alias("Long_Term_Cycle_Cos"),
    )

    # Add expected listening time based on sentiment
    df = df.with_columns(
        (pl.col("Episode_Length_minutes") * pl.col("Sentiment_Multiplier")).alias("Expected_Listening_Time_Sentiment"),
        ((pl.col("Episode_Length_minutes") + pl.col("Host_Popularity_percentage") + pl.col("Guest_Popularity_percentage")) / (3.0)).alias(
            "Mean_Important_Features"
        ),
        (
            pl.col("Episode_Length_minutes") * 0.5
            + pl.col("Number_of_Ads") * 0.2
            + pl.col("Host_Popularity_percentage") * 0.15
            + pl.col("Guest_Popularity_percentage") * 0.15
        ).alias("Episode_Complexity_Score"),
    )

    important_cols = [
        "Episode_Length_minutes",
        "Host_Popularity_percentage",
        "Guest_Popularity_percentage",
        "Number_of_Ads",
        "ELen_Int",
        "ELen_Dec",
        "HPperc_Int",
        "HPperc_Dec",
    ]
    interaction_transforms = []

    for i, col1 in enumerate(important_cols):
        for col2 in important_cols[i + 1 :]:
            interaction_transforms.append((pl.col(col1) * pl.col(col2)).alias(f"{col1}_{col2}_mult"))
            interaction_transforms.append((pl.col(col1) + pl.col(col2)).alias(f"{col1}_{col2}_sum"))
            interaction_transforms.append((pl.col(col1) - pl.col(col2)).alias(f"{col1}_{col2}_diff"))
            interaction_transforms.append(pl.when(pl.col(col2).abs() > 1e-10).then(pl.col(col1) / pl.col(col2)).otherwise(0).alias(f"{col1}_{col2}_div"))

            interaction_transforms.append((0.7 * pl.col(col1) + 0.3 * pl.col(col2)).alias(f"{col1}_{col2}_wgt_avg1"))
            interaction_transforms.append((0.3 * pl.col(col1) + 0.7 * pl.col(col2)).alias(f"{col1}_{col2}_wgt_avg2"))

            interaction_transforms.append(
                pl.when((pl.col(col1) >= 0) & (pl.col(col2) >= 0)).then((pl.col(col1) * pl.col(col2)).sqrt()).otherwise(0).alias(f"{col1}_{col2}_geo_mean")
            )
            interaction_transforms.append(
                pl.when((pl.col(col1).abs() > 1e-10) & (pl.col(col2).abs() > 1e-10))
                .then(2 / (1 / pl.col(col1) + 1 / pl.col(col2)))
                .otherwise(0)
                .alias(f"{col1}_{col2}_harm_mean")
            )

            interaction_transforms.append(pl.max_horizontal(pl.col(col1), pl.col(col2)).alias(f"{col1}_{col2}_max"))
            interaction_transforms.append(pl.min_horizontal(pl.col(col1), pl.col(col2)).alias(f"{col1}_{col2}_min"))

            interaction_transforms.append(
                (pl.max_horizontal(pl.col(col1), pl.col(col2)) - pl.min_horizontal(pl.col(col1), pl.col(col2))).alias(f"{col1}_{col2}_range")
            )
            interaction_transforms.append((pl.col(col1) - pl.col(col2)).pow(2).alias(f"{col1}_{col2}_sq_diff"))

            interaction_transforms.append(
                pl.when(pl.max_horizontal(pl.col(col1).abs(), pl.col(col2).abs()) > 1e-10)
                .then(((pl.col(col1) - pl.col(col2)).abs() / pl.max_horizontal(pl.col(col1).abs(), pl.col(col2).abs())))
                .otherwise(0)
                .alias(f"{col1}_{col2}_pct_diff")
            )
            interaction_transforms.append(
                pl.when((pl.col(col1) > 0) & (pl.col(col2) > 0))
                .then((pl.col(col1) + 1).log() * (pl.col(col2) + 1).log())
                .otherwise(0)
                .alias(f"{col1}_{col2}_log_prod")
            )

            interaction_transforms.append((pl.col(col1).pow(2) * pl.col(col2)).alias(f"{col1}_sq_{col2}"))
            interaction_transforms.append((pl.col(col1) * pl.col(col2).pow(2)).alias(f"{col1}_{col2}_sq"))

            interaction_transforms.append(
                pl.when(pl.col(col2).pow(2).abs() > 1e-10).then(pl.col(col1).pow(2) / pl.col(col2).pow(2)).otherwise(0).alias(f"{col1}_sq_{col2}_sq_ratio")
            )
            interaction_transforms.append((pl.col(col1).tanh() * pl.col(col2).tanh()).alias(f"{col1}_{col2}_tanh_prod"))

    df = df.with_columns(interaction_transforms)

    # Convert columns to categorical
    for col in ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Num"]:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))

    return df


def get_combinations(df, columns_to_encode, pair_sizes):
    df_length = len(df)

    target_ratios = []
    # target_ratios.extend(np.arange(0.01, 0.3, 0.05).tolist())
    # target_ratios.extend(np.arange(0.3, 1.01, 0.008).tolist())
    target_ratios.extend(np.arange(0.001, 0.999, 0.05).tolist())

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


def cols_encode(df, combinations_list):
    batch_size = 20
    for i in range(0, len(combinations_list), batch_size):
        batch = combinations_list[i : i + batch_size]

        for cols in tqdm(batch):
            new_col_name = "colen_" + "_".join(cols)
            concat_expr = pl.col(cols[0]).cast(pl.Utf8)

            for col_name in cols[1:]:
                concat_expr = concat_expr + "_" + pl.col(col_name).cast(pl.Utf8)

            df = df.with_columns(concat_expr.alias(new_col_name).cast(pl.Categorical))

        gc.collect()

        mem_usage = sum(df.estimated_size() for col in df.columns) / (1024 * 1024)
        print(f"Memory usage: {mem_usage:.2f} MB")
        print(f"Total number of columns: {len(df.columns)}")

    return df


def encode_target(X_train, X_valid, encode_columns, target, random_state=cfg.random_state):
    """
    Encode columns using TargetEncoder

    Args:
        X_train: Training dataframe
        X_valid: Validation dataframe
        encode_columns: List of column names to encode
        target: Target values as a polars Series
        random_state: Random state for reproducibility

    Returns:
        X_train, X_valid with encoded columns added
    """
    from sklearn.preprocessing import TargetEncoder

    # Convert target to numpy if it's a polars Series
    if isinstance(target, pl.Series):
        target_values = target.to_numpy()
    else:
        target_values = target

    # Create encoder
    encoder = TargetEncoder(random_state=random_state)

    # Ensure encode_columns is a list
    if isinstance(encode_columns, str):
        encode_columns = [encode_columns]

    for col in encode_columns:
        # Extract column to encode
        X_train_col = X_train[col].to_numpy().reshape(-1, 1)
        X_valid_col = X_valid[col].to_numpy().reshape(-1, 1)

        # Fit and transform
        encoded_train = encoder.fit_transform(X_train_col, target_values)
        encoded_valid = encoder.transform(X_valid_col)

        # Add encoded column back to dataframes
        encoded_col_name = f"{col}_{target.name}_encoded"

        X_train = X_train.with_columns(pl.Series(encoded_col_name, encoded_train.flatten()))
        X_valid = X_valid.with_columns(pl.Series(encoded_col_name, encoded_valid.flatten()))

    return X_train, X_valid


def get_dfs(cfg=cfg):
    # Read CSV files using polars
    df_train = pl.read_csv(cfg.train_path)
    df_train = df_train.filter(pl.col("Number_of_Ads").is_not_null())
    df_test = pl.read_csv(cfg.test_path)

    df_train = df_train.drop("id")
    df_test = df_test.drop("id")

    target_col = "Listening_Time_minutes"
    y_train = df_train[target_col]
    df_train = df_train.drop(target_col)

    # Do train/test split on numpy arrays
    X_train, X_valid, y_train, y_valid = train_test_split(
        df_train,
        y_train,
        test_size=0.2,  # 20% for validation
        random_state=cfg.random_state,
    )

    # Merge with df_podcast
    df_pltpd = pl.read_csv(cfg.pltpd_path)
    df_pltpd = df_pltpd.filter(pl.col("Listening_Time_minutes").is_not_null())

    # Extract target column and prepare for concatenation
    y_train_pltpd = df_pltpd["Listening_Time_minutes"]
    df_pltpd = df_pltpd.drop("Listening_Time_minutes")
    df_pltpd = df_pltpd.with_columns(pl.col("Number_of_Ads").cast(pl.Float64))

    # Now concatenate with matching schemas
    X_train = pl.concat([X_train, df_pltpd], how="vertical")
    y_train = pl.concat([y_train, y_train_pltpd], how="vertical")

    # Preprocess dataframes
    X_train = preprocess(X_train)
    X_valid = preprocess(X_valid, X_train)

    # Create combined df_train for feature engineering
    df_train = X_train.with_columns(y_train.alias("Listening_Time_minutes"))

    # Feature engineering
    X_train = feature_eng(X_train, df_train)
    X_valid = feature_eng(X_valid, df_train)

    # before_encode_len = len(X_train.columns)
    # columns_to_encode = [
    #     "Host_Popularity_percentage",
    #     "Guest_Popularity_percentage",
    #     "Episode_Length_minutes",
    #     "Episode_Num",
    #     "Podcast_Name",
    #     "Publication_Day",
    #     "Publication_Time",
    #     "Episode_Sentiment",
    #     "Genre",
    #     "Number_of_Ads",
    #     "Episode_Length_minutes_NaN",
    #     "Guest_Popularity_percentage_NaN",
    #     "HPperc_Int",
    #     "HPperc_Dec",
    #     "ELen_Int",
    #     "ELen_Dec",
    #     "Length_per_Ads",
    # ]
    # # pair_size = [2, 3, 4]
    # # pair_size = [2, 3]
    # pair_size = [2, 3]
    # combinations_list = get_combinations(X_train, columns_to_encode, pair_size)
    # print("Combinations list length:", len(combinations_list))
    # print("Combinations list:", combinations_list)

    # X_train = cols_encode(X_train, combinations_list)
    # X_valid = cols_encode(X_valid, combinations_list)

    # encoded_columns = X_train.columns[before_encode_len:]
    # print("Length of train columns:", before_encode_len)

    # X_train, X_valid = encode_target(X_train, X_valid, encoded_columns, y_train)

    # encoded_columns = [col for col in encoded_columns if col != "Episode_Length_minutes"]
    # X_train, X_valid = encode_target(X_train, X_valid, encoded_columns, X_train["Episode_Length_minutes"])

    # X_train = X_train.drop(encoded_columns)
    # X_valid = X_valid.drop(encoded_columns)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
    }
