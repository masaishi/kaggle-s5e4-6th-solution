import gc
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import TargetEncoder
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
    df["Episode_Num"] = df["Episode_Title"].str[8:].astype(int)  # Convert to int before log transform
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
    df["Episode_Length_minutes_NaN"] = df["Episode_Length_minutes"].isna().astype(int).astype("category")
    df["Guest_Popularity_percentage_NaN"] = df["Guest_Popularity_percentage"].isna().astype(int).astype("category")

    # Replacing null values by median
    df["Episode_Length_minutes"].fillna(df["Episode_Length_minutes"].median(), inplace=True)
    df["Guest_Popularity_percentage"].fillna(df["Guest_Popularity_percentage"].median(), inplace=True)

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

    df["Length_per_Ads"] = (df["Episode_Length_minutes"] / (df["Number_of_Ads"] + 1)).fillna(0)
    df["Length_per_Host"] = (df["Episode_Length_minutes"] / (df["Host_Popularity_percentage"] + 1)).fillna(0)
    df["Length_per_Guest"] = (df["Episode_Length_minutes"] / (df["Guest_Popularity_percentage"] + 1)).fillna(0)

    df["Podcast_Name"] = df["Podcast_Name"].astype("category")
    df["Genre"] = df["Genre"].astype("category")
    df["Publication_Day"] = df["Publication_Day"].astype("category")
    df["Publication_Time"] = df["Publication_Time"].astype("category")
    df["Episode_Sentiment"] = df["Episode_Sentiment"].astype("category")
    df["Episode_Num"] = df["Episode_Num"].astype("category")

    return df


def cols_encode(df):
    # Order of importance
    columns_to_encode = [
        "Host_Popularity_percentage",
        "Guest_Popularity_percentage",
        "Episode_Length_minutes",
        "Episode_Num",
        "Podcast_Name",
        "Publication_Day",
        "Publication_Time",
        "Episode_Sentiment",
        "Genre",
        "Number_of_Ads",
        "Episode_Length_minutes_NaN",
        "Guest_Popularity_percentage_NaN",
    ]

    pair_size = [2]

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

            print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            print(f"Total number of columns: {len(df.columns)}")

        print("=" * 20)

    columns_to_encode = [
        "Host_Popularity_percentage",
        "Guest_Popularity_percentage",
        "Episode_Length_minutes",
        "Episode_Num",
        "Podcast_Name",
        "Publication_Day",
        "Guest_Popularity_percentage_NaN",
    ]

    pair_size = [3, 4]

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

            print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
            print(f"Total number of columns: {len(df.columns)}")

        print("=" * 20)

    return df


def get_dfs(cfg=cfg):
    df_train = pd.read_csv(cfg.train_path, index_col="id")
    df_test = pd.read_csv(cfg.test_path, index_col="id")

    target_col = "Listening_Time_minutes"
    y_train = df_train[target_col].copy()
    df_train = df_train.drop(columns=[target_col])

    # Split to df_train and df_val
    X_train, X_valid, y_train, y_valid = train_test_split(
        df_train,
        y_train,
        test_size=0.2,  # 20% for validation
        random_state=42,
    )
    X_test = df_test.copy()

    # Merge with df_podcast
    df_pltpd = pd.read_csv(cfg.pltpd_path)
    df_pltpd = df_pltpd.dropna(subset=["Listening_Time_minutes"])
    df_pltpd = df_pltpd.reset_index(drop=True)
    df_pltpd.index = df_pltpd.index + 1000000
    y_train = pd.concat([y_train, df_pltpd["Listening_Time_minutes"]], axis=0).reset_index(drop=True)
    df_pltpd = df_pltpd.drop(columns=["Listening_Time_minutes"])
    X_train = pd.concat([X_train, df_pltpd], axis=0)

    # # Sample 100 fors X_train and y_train
    # X_train = X_train.sample(100, random_state=42)
    # y_train = y_train.sample(100, random_state=42)

    X_train = preprocess(X_train)
    X_valid = preprocess(X_valid)
    X_test = preprocess(X_test)

    X_desc = X_train.describe()
    X_train = feature_eng(X_train, X_desc)
    X_valid = feature_eng(X_valid, X_desc)
    X_test = feature_eng(X_test, X_desc)

    before_encode_len = len(X_train.columns)
    print("Length of train columns:", before_encode_len)

    X_train = cols_encode(X_train)
    X_valid = cols_encode(X_valid)
    X_test = cols_encode(X_test)

    X_test = X_test[X_train.columns].copy()

    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

    # Target encoding
    print("Before encoding columns:", before_encode_len)
    encoded_columns = X_train.columns[before_encode_len:]

    from sklearn.preprocessing import TargetEncoder

    encoder = TargetEncoder(random_state=cfg.random_state)
    X_train[encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
    X_valid[encoded_columns] = encoder.transform(X_valid[encoded_columns])
    X_test[encoded_columns] = encoder.transform(X_test[encoded_columns])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
    }
