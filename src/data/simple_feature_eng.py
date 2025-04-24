import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.metrics import mean_squared_error


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
    numeric_cols = df_train.select(cs.numeric()).columns
    stats = {
        col: {
            "mean": df_train.select(pl.col(col).mean()).item(),
            "std": df_train.select(pl.col(col).std()).item(),
        }
        for col in numeric_cols
    }

    transformations = []
    for col in numeric_cols:
        transformations.append(((pl.col(col) - stats[col]["mean"]) / stats[col]["std"]).alias(f"{col}"))
    df = df.with_columns(transformations)

    # Add std and mean of Listening_Time_minutes
    df = df.with_columns(
        (stats["Listening_Time_minutes"]["mean"]).alias("Listening_Time_minutes_mean"),
        (stats["Listening_Time_minutes"]["std"]).alias("Listening_Time_minutes_std"),
    )

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

    # Convert columns to categorical
    for col in ["Podcast_Name", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Num"]:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical))

    return df
