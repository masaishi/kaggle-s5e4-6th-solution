import gc
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
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


def preprocess(df):
    df["Episode_Num"] = (
        df["Episode_Title"].str[8:].astype(int)
    )  # Convert to int before log transform
    df = df.drop(columns=["Episode_Title"])

    # Convert categorical variables
    df["Genre"] = df["Genre"].replace(re_dict["genr_dict"])
    df["Podcast_Name"] = df["Podcast_Name"].replace(re_dict["podc_dict"])
    df["Publication_Day"] = df["Publication_Day"].replace(re_dict["week_dict"])
    df["Publication_Time"] = df["Publication_Time"].replace(re_dict["time_dict"])
    df["Episode_Sentiment"] = df["Episode_Sentiment"].replace(re_dict["sent_dict"])

    df.loc[df["Episode_Length_minutes"] > 121.0, "Episode_Length_minutes"] = 121.0
    df.loc[df["Number_of_Ads"] > 103.91, "Number_of_Ads"] = 103.91

    # Define categorical columns
    df["Episode_Length_minutes_NaN"] = (
        df["Episode_Length_minutes"].isna().astype(int).astype("category")
    )
    df["Guest_Popularity_percentage_NaN"] = (
        df["Guest_Popularity_percentage"].isna().astype(int).astype("category")
    )

    # Replacing null values by median
    df["Episode_Length_minutes"].fillna(
        df["Episode_Length_minutes"].median(), inplace=True
    )
    df["Guest_Popularity_percentage"].fillna(
        df["Guest_Popularity_percentage"].median(), inplace=True
    )

    return df


def feature_eng(df, df_desc=None):
    # Better capture cyclical nature of day and time
    df["Day_sin"] = np.sin(2 * np.pi * df["Publication_Day"] / 7)
    df["Day_cos"] = np.cos(2 * np.pi * df["Publication_Day"] / 7)
    df["Time_sin"] = np.sin(2 * np.pi * df["Publication_Time"] / 4)
    df["Time_cos"] = np.cos(2 * np.pi * df["Publication_Time"] / 4)

    # Higher frequency sinusoidal features for day and time
    df["Day_sin2"] = np.sin(4 * np.pi * df["Publication_Day"] / 7)
    df["Day_cos2"] = np.cos(4 * np.pi * df["Publication_Day"] / 7)
    df["Time_sin2"] = np.sin(4 * np.pi * df["Publication_Time"] / 24)
    df["Time_cos2"] = np.cos(4 * np.pi * df["Publication_Time"] / 24)

    df["Length_per_Ads"] = (
        df["Episode_Length_minutes"] / (df["Number_of_Ads"] + 1)
    ).fillna(0)

    groups = [
        "Podcast_Name",
        "Episode_Length_minutes_NaN",
        "Guest_Popularity_percentage_NaN",
        "Publication_Day",
        "Publication_Time",
        "Genre",
    ]
    for group in groups:
        numeric_cols = [
            "Episode_Num",
            "Episode_Length_minutes",
            "Number_of_Ads",
            "Host_Popularity_percentage",
            "Guest_Popularity_percentage",
        ]
        for col in numeric_cols:
            df[f"{group}_{col}_norm"] = df.groupby(group)[col].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            )

    df["Podcast_Name"] = df["Podcast_Name"].astype("category")
    df["Genre"] = df["Genre"].astype("category")
    df["Publication_Day"] = df["Publication_Day"].astype("category")
    df["Publication_Time"] = df["Publication_Time"].astype("category")
    df["Episode_Sentiment"] = df["Episode_Sentiment"].astype("category")
    df["Episode_Num"] = df["Episode_Num"].astype("category")

    return df


def cols_encode(df):
    columns_to_encode = [
        "Podcast_Name",
        "Episode_Num",
        "Episode_Length_minutes",
        # "Genre",
        "Host_Popularity_percentage",
        "Guest_Popularity_percentage",
        "Publication_Day",
        "Publication_Time",
        "Number_of_Ads",
        # "Episode_Sentiment",
        "Episode_Length_minutes_NaN",
        "Guest_Popularity_percentage_NaN",
        # "Podcast_Name_Episode_Num_norm",
    ]

    pair_size = [2, 3, 4]

    for r in pair_size:
        combinations_list = list(combinations(columns_to_encode, r))
        batch_size = 20

        print("\n pair_size:", r, "\n")

        for i in range(0, len(combinations_list), batch_size):
            batch = combinations_list[i : i + batch_size]

            for cols in tqdm(batch):
                new_col_name = "colen_" + "_".join(cols)
                df[new_col_name] = df[list(cols)].astype(str).agg("_".join, axis=1)
                df[new_col_name] = df[new_col_name].astype("category")

            gc.collect()

            print(
                f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
            )
            print(f"Total number of columns: {len(df.columns)}")

        print("=" * 20)
    return df


def get_dfs():
    df_train = pd.read_csv(cfg.train_path, index_col="id")
    df_test = pd.read_csv(cfg.test_path, index_col="id")
    df_sub = pd.read_csv(cfg.sub_path, index_col="id")
    df_podcast = pd.read_csv(cfg.podcast_path, index_col="id")

    is_dev_mode = False
    # is_dev_mode = True
    if is_dev_mode:
        # df_train = df_train.sample(10000, random_state=42)
        df_train = df_train.sample(100, random_state=42)
        df_test = df_test[:10]
        df_sub = df_sub[:10]

    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    target_col = "Listening_Time_minutes"
    y_train = df_train[target_col].copy()
    df_train = df_train.drop(columns=[target_col])

    df_desc = df_train.describe()

    df_train = feature_eng(df_train, df_desc)
    df_test = feature_eng(df_test, df_desc)

    before_encode_len = len(df_train.columns)
    print("Length of train columns:", before_encode_len)

    df_train = cols_encode(df_train)
    df_test = cols_encode(df_test)

    return df_train, df_test, df_sub, y_train, df_desc, before_encode_len
