import gc
import random
from itertools import combinations

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.preprocessing import TargetEncoder
from tqdm import tqdm

from config import cfg
from data.data_class import DatasetX

before_fe_selecteds = [
    "Podcast_Name-Host_Popularity_percentage-Length_per_Guest-Long_Term_Cycle_Cos",
    "Genre-Host_Popularity_percentage-Episode_Num-Length_per_Guest",
    "Podcast_Name-Publication_Day-Length_per_Host-Length_per_Guest",
    "Podcast_Name-Publication_Time-Length_per_Host-Length_per_Guest",
    "Host_Popularity_percentage-Publication_Day-Episode_Num-Length_per_Guest",
    "Host_Popularity_percentage-Number_of_Ads-Episode_Num-Length_per_Guest",
    "Host_Popularity_percentage-Publication_Time-Episode_Num-Length_per_Guest",
    "Guest_Popularity_percentage-Episode_Num-Time_cos-Length_per_Host",
    "Podcast_Name-Episode_Num-Length_per_Guest-HPperc_Int",
    "Host_Popularity_percentage-Episode_Num-Episode_Length_minutes_NaN-Length_per_Guest",
    "Podcast_Name-Publication_Day-Episode_Num-Length_per_Host",
    "Podcast_Name-Episode_Num-Length_per_Ads-Length_per_Host",
    "Podcast_Name-Publication_Time-Episode_Num-Length_per_Host",
    "Podcast_Name-Episode_Num-Time_sin-Length_per_Host",
    "Podcast_Name-Episode_Num-Length_per_Host-Is_Positive_Sentiment",
    "Podcast_Name-Episode_Num-Guest_Popularity_percentage_NaN-Length_per_Host",
    "Podcast_Name-Episode_Length_minutes-Host_Popularity_percentage-Episode_Num",
    "Podcast_Name-Host_Popularity_percentage-Episode_Num-ELen_Dec",
    "Genre-Episode_Num-Length_per_Host-Is_Positive_Sentiment",
    "Genre-Episode_Num-Guest_Popularity_percentage_NaN-Length_per_Host",
    "Podcast_Name-Publication_Day-Episode_Sentiment-Length_per_Host",
    "Publication_Day-Publication_Time-Episode_Num-Length_per_Host",
    "Publication_Day-Episode_Sentiment-Episode_Num-Length_per_Host",
    "Publication_Time-Episode_Num-Length_per_Ads-Length_per_Host",
    "Publication_Time-Number_of_Ads-Length_per_Host-Long_Term_Cycle_Cos",
    "Podcast_Name-Host_Popularity_percentage-Episode_Num-ELen_Int",
    "Podcast_Name-Host_Popularity_percentage-ELen_Int-Long_Term_Cycle_Sin",
    "Number_of_Ads-Episode_Num-Guest_Popularity_percentage_NaN-Length_per_Host",
    "Episode_Num-Guest_Popularity_percentage_NaN-Time_cos-Length_per_Host",
    "Podcast_Name-Host_Popularity_percentage-Publication_Day-Episode_Num",
    "Genre-Host_Popularity_percentage-Episode_Num-ELen_Int",
    "Time_cos-Length_per_Guest-HPperc_Int-Is_Positive_Sentiment",
    "Episode_Sentiment-Episode_Num-Length_per_Guest-ELen_Int",
    "Episode_Length_minutes-Publication_Time-Guest_Popularity_percentage-HPperc_Int",
    "Host_Popularity_percentage-Publication_Day-Episode_Num-ELen_Int",
    "Host_Popularity_percentage-Publication_Day-ELen_Int-Long_Term_Cycle_Sin",
    "Episode_Length_minutes_NaN-Time_sin-Length_per_Guest-HPperc_Int",
    "Host_Popularity_percentage-Episode_Num-Time_sin2-ELen_Int",
    "Host_Popularity_percentage-Episode_Sentiment-Episode_Num-ELen_Int",
    "Host_Popularity_percentage-Episode_Sentiment-ELen_Int-Long_Term_Cycle_Sin",
    "Host_Popularity_percentage-Episode_Num-ELen_Int-Is_Positive_Sentiment",
    "Host_Popularity_percentage-Episode_Num-Guest_Popularity_percentage_NaN-ELen_Int",
    "Host_Popularity_percentage-Guest_Popularity_percentage_NaN-ELen_Int-Long_Term_Cycle_Cos",
    "Episode_Length_minutes_NaN-Length_per_Guest-ELen_Int-HPperc_Int",
    "Host_Popularity_percentage-Episode_Num-Episode_Length_minutes_NaN-ELen_Int",
    "Host_Popularity_percentage-Episode_Length_minutes_NaN-ELen_Int-Long_Term_Cycle_Sin",
    "Episode_Length_minutes-Number_of_Ads-Episode_Num-Day_sin",
    "Episode_Length_minutes-Number_of_Ads-Day_cos-Long_Term_Cycle_Cos",
    "Episode_Length_minutes-Day_sin-Length_per_Ads-HPperc_Int",
    "Episode_Length_minutes-Publication_Day-Episode_Sentiment-Long_Term_Cycle_Sin",
    "Episode_Length_minutes-Number_of_Ads-Episode_Num-Time_sin2",
    "Publication_Time-Episode_Num-Length_per_Ads-ELen_Dec",
    "Episode_Length_minutes-Time_cos2-Length_per_Ads-Long_Term_Cycle_Sin",
    "Episode_Length_minutes-Publication_Time-Length_per_Ads-HPperc_Int",
    "Episode_Num-Time_cos-Length_per_Ads-ELen_Dec",
    "Number_of_Ads-Episode_Sentiment-Length_per_Ads-HPperc_Int",
    "Episode_Length_minutes-Number_of_Ads-Time_cos-HPperc_Int",
    "Time_sin-Length_per_Ads-ELen_Int-HPperc_Int",
    "Episode_Length_minutes-Number_of_Ads-Episode_Num-Sentiment_Multiplier",
    "Number_of_Ads-Length_per_Ads-Is_Positive_Sentiment-Long_Term_Cycle_Cos",
    "Episode_Length_minutes-Number_of_Ads-HPperc_Int-Is_Positive_Sentiment",
    "Guest_Popularity_percentage_NaN-Length_per_Ads-ELen_Int-Long_Term_Cycle_Cos",
    "Number_of_Ads-Guest_Popularity_percentage_NaN-Length_per_Ads-HPperc_Int",
    "Episode_Length_minutes-Time_cos-HPperc_Int-Is_Positive_Sentiment",
    "Episode_Length_minutes-Episode_Num-Episode_Length_minutes_NaN-Length_per_Ads",
    "Episode_Length_minutes-Episode_Length_minutes_NaN-Length_per_Ads-Long_Term_Cycle_Cos",
    "Episode_Length_minutes-Number_of_Ads-HPperc_Int",
    "Episode_Length_minutes-Episode_Sentiment-Episode_Num",
    "Episode_Length_minutes-Episode_Num-Episode_Length_minutes_NaN-Time_sin",
    "Episode_Length_minutes-Episode_Sentiment-Episode_Length_minutes_NaN-HPperc_Int",
    "Episode_Length_minutes-Time_cos-HPperc_Int",
    "Episode_Length_minutes-Episode_Num-Episode_Length_minutes_NaN-Is_Positive_Sentiment",
    "Episode_Length_minutes-Episode_Length_minutes_NaN-Is_Positive_Sentiment-Long_Term_Cycle_Cos",
    "Episode_Length_minutes-Episode_Length_minutes_NaN-HPperc_Int-Is_Positive_Sentiment",
    "Episode_Length_minutes-Guest_Popularity_percentage_NaN-ELen_Dec-HPperc_Int",
    "Publication_Time-Number_of_Ads-Day_sin2-Length_per_Ads",
    "Episode_Length_minutes-Episode_Num-Episode_Length_minutes_NaN-ELen_Dec",
    "Episode_Length_minutes-Number_of_Ads-Episode_Sentiment-Day_sin2",
    "Episode_Length_minutes-Publication_Day-Number_of_Ads-Sentiment_Multiplier",
    "Genre-Number_of_Ads-ELen_Int-HPperc_Int",
    "ELen_Dec-Is_Positive_Sentiment",
    "Number_of_Ads-Guest_Popularity_percentage_NaN-ELen_Int",
    "Number_of_Ads-Guest_Popularity_percentage_NaN-ELen_Int-Is_Positive_Sentiment",
    "Number_of_Ads-Episode_Sentiment-Guest_Popularity_percentage_NaN-ELen_Int",
    "Publication_Time-Number_of_Ads-Guest_Popularity_percentage_NaN-ELen_Int",
    "Publication_Day-Number_of_Ads-Guest_Popularity_percentage_NaN-ELen_Int",
    "Genre-Number_of_Ads-Guest_Popularity_percentage_NaN-ELen_Int",
    "Time_cos-Length_per_Guest-HPperc_Dec-Long_Term_Cycle_Sin",
    "Podcast_Name-Publication_Day-Length_per_Guest-HPperc_Int",
    "Genre-Publication_Day-Length_per_Guest-HPperc_Int",
    "Host_Popularity_percentage-Publication_Time-Guest_Popularity_percentage-ELen_Dec",
    "Host_Popularity_percentage-Guest_Popularity_percentage-Time_cos-Long_Term_Cycle_Cos",
    "Day_cos-Length_per_Ads-HPperc_Dec-Long_Term_Cycle_Cos",
    "Podcast_Name-Episode_Length_minutes-Episode_Length_minutes_NaN-Length_per_Host",
    "Episode_Length_minutes-Genre-Number_of_Ads-HPperc_Dec",
    "Episode_Sentiment-Time_cos-Length_per_Ads-Length_per_Host",
    "Publication_Day-Publication_Time-Length_per_Ads-Long_Term_Cycle_Cos",
    "Episode_Length_minutes-Guest_Popularity_percentage_NaN-HPperc_Int-Long_Term_Cycle_Sin",
    "Guest_Popularity_percentage-Episode_Sentiment-Time_cos-Length_per_Ads",
    "Genre-Host_Popularity_percentage-Time_cos-ELen_Int",
    "Host_Popularity_percentage-Publication_Day-Time_cos-ELen_Int",
    "Host_Popularity_percentage-Publication_Day-Time_sin-ELen_Int",
    "Guest_Popularity_percentage-Number_of_Ads-Episode_Length_minutes_NaN-ELen_Int",
    "Guest_Popularity_percentage-Episode_Length_minutes_NaN-Time_sin-ELen_Int",
    "Episode_Length_minutes-Number_of_Ads-Guest_Popularity_percentage_NaN-Is_Positive_Sentiment",
    "Episode_Length_minutes-Number_of_Ads-Episode_Sentiment-Episode_Length_minutes_NaN",
    "Episode_Length_minutes-Publication_Day-Guest_Popularity_percentage_NaN-Is_Positive_Sentiment",
    "Guest_Popularity_percentage_NaN-ELen_Int-HPperc_Int-HPperc_Dec",
    "Podcast_Name-Time_sin-ELen_Int-HPperc_Int",
    "Publication_Day-Number_of_Ads-ELen_Int-HPperc_Int",
    "ELen_Int-Is_Positive_Sentiment",
    "Episode_Sentiment-Episode_Length_minutes_NaN-ELen_Int",
    "Publication_Day-Episode_Length_minutes_NaN-ELen_Int",
    "Number_of_Ads-ELen_Dec",
    "Publication_Day-Time_sin-ELen_Int",
    "Podcast_Name-Publication_Time-Length_per_Guest-HPperc_Dec",
    "Genre-Publication_Time-Length_per_Guest-HPperc_Dec",
    "Podcast_Name-Host_Popularity_percentage-Guest_Popularity_percentage-Time_sin2",
    "Guest_Popularity_percentage-Episode_Num-ELen_Dec-HPperc_Int",
    "Podcast_Name-Genre-Guest_Popularity_percentage-Episode_Num",
    "Genre-Guest_Popularity_percentage-Episode_Sentiment-Episode_Num",
    "Genre-Guest_Popularity_percentage_NaN-Length_per_Host-Is_Positive_Sentiment",
    "Podcast_Name-Host_Popularity_percentage-ELen_Dec-HPperc_Int",
    "Podcast_Name-Episode_Length_minutes-Episode_Sentiment-Long_Term_Cycle_Sin",
    "Publication_Day-Episode_Sentiment-Length_per_Host-HPperc_Dec",
    "Publication_Time-Length_per_Ads-HPperc_Int-Long_Term_Cycle_Cos",
    "Host_Popularity_percentage-Publication_Time-Day_sin2-Long_Term_Cycle_Cos",
    "Episode_Length_minutes_NaN-Guest_Popularity_percentage_NaN-Length_per_Host-Is_Positive_Sentiment",
    "Genre-Episode_Num-ELen_Int-HPperc_Dec",
    "Episode_Sentiment-Episode_Length_minutes_NaN-Guest_Popularity_percentage_NaN-Length_per_Guest",
    "Episode_Length_minutes_NaN-Time_sin-Time_cos-Length_per_Guest",
    "Number_of_Ads-Episode_Num-ELen_Int-HPperc_Dec",
    "Number_of_Ads-ELen_Int-HPperc_Dec-Long_Term_Cycle_Sin",
    "Episode_Num-Time_sin-ELen_Int-HPperc_Dec",
    "Podcast_Name-Time_cos2-ELen_Int-HPperc_Dec",
    "Episode_Length_minutes_NaN-Time_sin-Length_per_Ads-Is_Positive_Sentiment",
    "Guest_Popularity_percentage_NaN-Time_cos-Length_per_Ads-Sentiment_Multiplier",
    "Publication_Day-Publication_Time-Guest_Popularity_percentage_NaN-Length_per_Ads",
    "Day_sin-Time_cos-Length_per_Ads-Is_Positive_Sentiment",
    "Publication_Day-Publication_Time-Episode_Length_minutes_NaN-Length_per_Ads",
    "Number_of_Ads-Time_cos-ELen_Dec",
    "Publication_Time-Number_of_Ads-ELen_Dec",
    "Publication_Day-Episode_Sentiment-ELen_Dec",
    "Publication_Day-Number_of_Ads-ELen_Dec",
    "Publication_Time-Number_of_Ads-ELen_Dec-Is_Positive_Sentiment",
    "Podcast_Name-Length_per_Guest-HPperc_Dec-Is_Positive_Sentiment",
    "Publication_Time-Number_of_Ads-Episode_Sentiment-ELen_Dec",
    "Guest_Popularity_percentage-Episode_Num-ELen_Dec-HPperc_Dec",
    "Publication_Day-Number_of_Ads-Time_sin-ELen_Dec",
    "Podcast_Name-Guest_Popularity_percentage-Episode_Num-ELen_Dec",
    "Podcast_Name-Guest_Popularity_percentage-ELen_Dec-HPperc_Int",
    "Episode_Num-Time_sin-Length_per_Ads-HPperc_Dec",
    "Episode_Sentiment-Length_per_Ads-HPperc_Dec-Long_Term_Cycle_Cos",
    "Podcast_Name-Episode_Num-Day_sin-Length_per_Ads",
    "Podcast_Name-Genre-Length_per_Ads-Long_Term_Cycle_Cos",
    "Publication_Time-Length_per_Host-ELen_Int-Is_Positive_Sentiment",
    "Length_per_Ads-HPperc_Int-Is_Positive_Sentiment-Long_Term_Cycle_Cos",
    "Host_Popularity_percentage-Episode_Num-Day_sin-Time_sin",
    "Length_per_Ads-Is_Positive_Sentiment",
    "Host_Popularity_percentage-Number_of_Ads-Episode_Sentiment-ELen_Dec",
    "Host_Popularity_percentage-Number_of_Ads-ELen_Dec-Is_Positive_Sentiment",
    "Host_Popularity_percentage-Number_of_Ads-Guest_Popularity_percentage_NaN-ELen_Dec",
    "Number_of_Ads-Episode_Num-ELen_Dec-HPperc_Int",
    "Host_Popularity_percentage-Number_of_Ads-Episode_Length_minutes_NaN-ELen_Dec",
    "Host_Popularity_percentage-Episode_Length_minutes_NaN-Time_cos-ELen_Dec",
    "ELen_Dec-HPperc_Int-Is_Positive_Sentiment-Long_Term_Cycle_Cos",
    "Host_Popularity_percentage-Episode_Length_minutes_NaN-Guest_Popularity_percentage_NaN-ELen_Dec",
    "Number_of_Ads-Time_cos-ELen_Dec-HPperc_Int",
    "Number_of_Ads-Time_sin2-ELen_Dec-HPperc_Int",
    "Publication_Time-Number_of_Ads-ELen_Dec-Long_Term_Cycle_Cos",
]

selecteds = [
    "ELen_Int-Is_Positive_Sentiment",
    "ELen_Dec",
    "ELen_Int-pte_Podcast_Name_Guest_Popularity_percentage_ELen_Dec_HPperc_Int-pte_Number_of_Ads",
    "Publication_Day-ELen_Int-pte_Podcast_Name_Guest_Popularity_percentage_ELen_Dec_HPperc_Int",
    "Publication_Day-ELen_Int-pte_Host_Popularity_percentage_Publication_Time_Episode_Num_Length_per_Guest",
    "Publication_Day-ELen_Int-pte_Genre_Episode_Num_Length_per_Host_Is_Positive_Sentiment",
    "Episode_Sentiment-pte_Genre_Episode_Num_Length_per_Host_Is_Positive_Sentiment-pte_ELen_Dec",
    "Genre-ELen_Int-pte_Podcast_Name_Publication_Time_Episode_Num_Length_per_Host",
    "Genre-ELen_Int-pte_Genre_Episode_Num_Length_per_Host_Is_Positive_Sentiment",
    "ELen_Dec-pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Number_of_Ads",
    "Publication_Day-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec-pte_ELen_Dec",
    "Episode_Length_minutes_NaN-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Podcast_Name_Guest_Popularity_percentage_ELen_Dec_HPperc_Int",
    "pte_Host_Popularity_percentage_Publication_Time_Episode_Num_Length_per_Guest-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Podcast_Name_Guest_Popularity_percentage_ELen_Dec_HPperc_Int",
    "pte_Podcast_Name_Publication_Time_Episode_Num_Length_per_Host-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int",
    "pte_Host_Popularity_percentage_Publication_Time_Episode_Num_Length_per_Guest-pte_Podcast_Name_Publication_Time_Episode_Num_Length_per_Host-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int",
    "Publication_Day-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_ELen_Dec",
    "pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Podcast_Name_Guest_Popularity_percentage_ELen_Dec_HPperc_Int",
    "pte_Podcast_Name_Publication_Time_Episode_Num_Length_per_Host-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec",
    "pte_Host_Popularity_percentage_Publication_Time_Episode_Num_Length_per_Guest-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num",
    "pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec",
    "pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num",
    "pte_Host_Popularity_percentage_Publication_Time_Episode_Num_Length_per_Guest-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Guest",
    "pte_Host_Popularity_percentage_Publication_Time_Episode_Num_Length_per_Guest-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int",
    "pte_Genre_Episode_Num_Length_per_Host_Is_Positive_Sentiment-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "pte_Host_Popularity_percentage_Publication_Time_Episode_Num_Length_per_Guest-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "Episode_Length_minutes_NaN-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "Guest_Popularity_percentage_NaN-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "Publication_Time-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "Guest_Popularity_percentage_NaN-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int",
    "pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int-pte_Length_per_Host",
    "pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num-pte_Length_per_Host",
    "pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Podcast_Name_Guest_Popularity_percentage_ELen_Dec_HPperc_Int-pte_Is_Positive_Sentiment",
    "Is_Positive_Sentiment-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int",
    "pte_Genre_Episode_Num_Length_per_Host_Is_Positive_Sentiment-pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Episode_Length_minutes",
    "pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec-pte_Episode_Length_minutes",
    "Podcast_Name-ELen_Int-pte_Length_per_Host",
    "Is_Positive_Sentiment-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int",
    "Episode_Length_minutes_NaN-pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int-pte_Episode_Length_minutes",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Is_Positive_Sentiment",
    "Is_Positive_Sentiment-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "pte_Length_per_Guest-pte_Episode_Length_squared",
    "pte_Podcast_Name_Publication_Time_Episode_Num_Length_per_Host-pte_Episode_Length_minutes-pte_Length_per_Guest",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Is_Positive_Sentiment",
    "pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Is_Positive_Sentiment",
    "pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Episode_Length_minutes-pte_Length_per_Guest",
    "pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Episode_Length_minutes-pte_Length_per_Guest",
    "Episode_Length_minutes_NaN-Expected_Listening_Time_Sentiment-pte_Podcast_Name_Publication_Time_Episode_Num_Length_per_Host",
    "pte_Genre_Episode_Num_Length_per_Host_Is_Positive_Sentiment-pte_Episode_Length_minutes-pte_Length_per_Host",
    "pte_Host_Popularity_percentage_Publication_Time_Episode_Num_Length_per_Guest-pte_Episode_Length_minutes-pte_Length_per_Host",
    "pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host-pte_Is_Positive_Sentiment",
    "pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num-pte_Episode_Length_minutes-pte_Length_per_Host",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num-pte_Episode_Length_minutes",
    "Episode_Sentiment-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int",
    "pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int-pte_Episode_Length_minutes-pte_Length_per_Host",
    "Episode_Sentiment-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Guest",
    "pte_Length_per_Host-pte_Length_per_Guest-pte_Episode_Length_squared",
    "pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Episode_Length_minutes-pte_Length_per_Host",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Length_per_Guest-pte_Episode_Length_squared",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Episode_Length_minutes-pte_Length_per_Host",
    "pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Episode_Length_minutes",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec-pte_Episode_Length_minutes",
    "pte_Length_per_Host-pte_Expected_Listening_Time_Sentiment",
    "ELen_Int-pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num-pte_Episode_Length_minutes",
    "Episode_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int",
    "Episode_Length_minutes_NaN-Expected_Listening_Time_Sentiment-pte_Length_per_Host",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Episode_Length_minutes",
    "ELen_Int-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec-pte_Episode_Length_minutes",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Episode_Length_minutes-pte_Length_per_Host",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Episode_Length_minutes-pte_Length_per_Guest",
    "pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Length_per_Host-pte_Expected_Listening_Time_Sentiment",
    "ELen_Int-HPperc_Int-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment",
    "ELen_Int-HPperc_Int-pte_Length_per_Host",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Length_per_Host-pte_Expected_Listening_Time_Sentiment",
    "Episode_Length_minutes_NaN-Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Expected_Listening_Time_Sentiment",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Length_per_Host-pte_Expected_Listening_Time_Sentiment",
    "ELen_Dec-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Episode_Length_minutes",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Length_per_Guest-pte_Expected_Listening_Time_Sentiment",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_ELen_Int-pte_HPperc_Int",
    "pte_Episode_Length_minutes-pte_Length_per_Guest-pte_ELen_Dec",
    "ELen_Dec-pte_Episode_Length_minutes-pte_Length_per_Guest",
    "ELen_Int-HPperc_Int-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int",
    "ELen_Int-pte_Episode_Length_minutes-pte_Length_per_Host",
    "pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_ELen_Int-pte_HPperc_Int",
    "ELen_Int-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_HPperc_Int",
    "Episode_Length_minutes-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec-pte_Length_per_Host",
    "Episode_Length_minutes-pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num-pte_Length_per_Host",
    "Episode_Length_minutes-pte_Length_per_Host-pte_Length_per_Guest",
    "Episode_Length_minutes-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Length_per_Host",
    "Episode_Length_minutes-pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int",
    "Episode_Length_minutes-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec",
    "Episode_Length_minutes-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int",
    "Episode_Length_minutes-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Length_per_Host",
    "ELen_Int-Expected_Listening_Time_Sentiment-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int",
    "Is_Positive_Sentiment-Expected_Listening_Time_Sentiment-pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int",
    "ELen_Int-Expected_Listening_Time_Sentiment-pte_Length_per_Guest",
    "Is_Positive_Sentiment-Expected_Listening_Time_Sentiment-pte_Length_per_Guest",
    "ELen_Int-Expected_Listening_Time_Sentiment-pte_Length_per_Host",
    "Is_Positive_Sentiment-Expected_Listening_Time_Sentiment-pte_Length_per_Host",
    "Guest_Popularity_percentage_NaN-Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment",
    "Guest_Popularity_percentage_NaN-Expected_Listening_Time_Sentiment-pte_Length_per_Host",
    "Is_Positive_Sentiment-Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int",
    "ELen_Int-pte_Length_per_Host-pte_Expected_Listening_Time_Sentiment",
    "pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int-pte_ELen_Int-pte_HPperc_Dec",
    "pte_Length_per_Host-pte_ELen_Dec-pte_Expected_Listening_Time_Sentiment",
    "ELen_Dec-pte_Length_per_Host-pte_Expected_Listening_Time_Sentiment",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_ELen_Int-pte_Expected_Listening_Time_Sentiment",
    "ELen_Int-Is_Positive_Sentiment-pte_ELen_Dec",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_ELen_Int-pte_HPperc_Dec",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_ELen_Dec-pte_Expected_Listening_Time_Sentiment",
    "ELen_Dec-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Expected_Listening_Time_Sentiment",
    "Publication_Day-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int-pte_Length_per_Host",
    "Episode_Length_minutes-Guest_Popularity_percentage_NaN-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec",
    "pte_Length_per_Host-pte_ELen_Int-pte_HPperc_Dec",
    "ELen_Dec-Expected_Listening_Time_Sentiment-pte_Genre_Episode_Num_Length_per_Host_Is_Positive_Sentiment",
    "ELen_Dec-Expected_Listening_Time_Sentiment-pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int",
    "ELen_Dec-Expected_Listening_Time_Sentiment-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec",
    "pte_Length_per_Host-pte_Episode_Length_squared-pte_Expected_Listening_Time_Sentiment",
    "Episode_Length_minutes-Guest_Popularity_percentage_NaN-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int",
    "Episode_Length_minutes-Guest_Popularity_percentage_NaN-pte_Length_per_Guest",
    "ELen_Dec-Expected_Listening_Time_Sentiment-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Episode_Length_minutes-pte_Expected_Listening_Time_Sentiment",
    "Expected_Listening_Time_Sentiment-pte_Podcast_Name_Publication_Time_Episode_Num_Length_per_Host-pte_Length_per_Guest",
    "Episode_Length_minutes-Guest_Popularity_percentage_NaN-pte_Length_per_Host",
    "Expected_Listening_Time_Sentiment-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int",
    "ELen_Dec-Expected_Listening_Time_Sentiment-pte_Length_per_Host",
    "Expected_Listening_Time_Sentiment-pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num-pte_Length_per_Host",
    "Guest_Popularity_percentage_NaN-Episode_Length_squared-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int",
    "Expected_Listening_Time_Sentiment-pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int-pte_Length_per_Host",
    "Expected_Listening_Time_Sentiment-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Length_per_Host",
    "Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Length_per_Host",
    "Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Length_per_Host",
    "Episode_Sentiment-Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int",
    "Is_Positive_Sentiment-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Episode_Length_minutes",
    "pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Episode_Length_minutes-pte_Number_of_Ads",
    "Genre-Guest_Popularity_percentage_NaN-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int",
    "ELen_Int-pte_Podcast_Name_Guest_Popularity_percentage_ELen_Dec_HPperc_Int-pte_Length_per_Ads",
    "ELen_Int-pte_Genre_Guest_Popularity_percentage_Episode_Sentiment_Episode_Num-pte_Length_per_Ads",
    "ELen_Int-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec-pte_Length_per_Ads",
    "pte_Podcast_Name_Publication_Time_Episode_Num_Length_per_Host-pte_Length_per_Ads-pte_ELen_Dec",
    "ELen_Int-Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier",
    "Publication_Day-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Publication_Time_Number_of_Ads_Guest_Popularity_percentage_NaN_ELen_Int",
    "ELen_Int-pte_Number_of_Ads_Episode_Num_ELen_Dec_HPperc_Int-pte_Length_per_Ads",
    "ELen_Int-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Length_per_Ads",
    "ELen_Int-pte_Length_per_Ads-pte_Length_per_Guest",
    "pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Length_per_Ads-pte_ELen_Dec",
    "ELen_Dec-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int-pte_Length_per_Ads",
    "ELen_Int-pte_Length_per_Ads-pte_Length_per_Host",
    "ELen_Int-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Length_per_Ads",
    "ELen_Dec-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Expected_Listening_Time_Sentiment",
    "pte_Length_per_Ads-pte_Length_per_Host-pte_ELen_Int",
    "pte_Length_per_Ads-pte_Length_per_Host-pte_ELen_Dec",
    "ELen_Int-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Length_per_Ads",
    "ELen_Dec-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Length_per_Ads",
    "Episode_Length_minutes-Episode_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int",
    "Episode_Length_minutes-Guest_Popularity_percentage_NaN-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier",
    "Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Podcast_Name_Guest_Popularity_percentage_ELen_Dec_HPperc_Int",
    "Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec",
    "Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Guest_Popularity_percentage_Number_of_Ads_Episode_Length_minutes_NaN_ELen_Int",
    "Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier",
    "Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Length_per_Host",
    "Episode_Length_minutes-pte_Length_per_Ads-pte_Length_per_Guest",
    "pte_Episode_Length_minutes-pte_Length_per_Ads-pte_Length_per_Host",
    "Episode_Length_minutes-pte_Length_per_Ads-pte_Length_per_Host",
    "Episode_Length_squared2-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int_Is_Positive_Sentiment-pte_Length_per_Ads",
    "Length_per_Ads-ELen_Dec-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec",
    "Number_of_Ads-Episode_Length_squared2-pte_Host_Popularity_percentage_Publication_Time_Guest_Popularity_percentage_ELen_Dec",
    "Episode_Length_minutes-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Length_per_Ads",
    "pte_Episode_Length_minutes-pte_Number_of_Ads-pte_Length_per_Host",
    "Episode_Length_minutes-Number_of_Ads-pte_Length_per_Host",
    "Number_of_Ads-Episode_Length_squared2-pte_Length_per_Host",
    "ELen_Int-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Length_per_Ads",
    "ELen_Dec-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Length_per_Ads",
    "Guest_Popularity_percentage_NaN-pte_Length_per_Ads-pte_ELen_Dec",
    "pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Episode_Length_minutes-pte_Length_per_Ads",
    "Episode_Length_minutes-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier-pte_Length_per_Ads",
    "Publication_Time-Episode_Length_squared2-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier",
    "Is_Positive_Sentiment-Expected_Listening_Time_Sentiment-pte_Number_of_Ads",
    "Publication_Time-Is_Positive_Sentiment-Expected_Listening_Time_Sentiment",
    "Number_of_Ads-Episode_Length_squared2-pte_Episode_Length_minutes_Publication_Day_Number_of_Ads_Sentiment_Multiplier",
    "ELen_Int-pte_Number_of_Ads-pte_Expected_Listening_Time_Sentiment",
    "Guest_Popularity_percentage_NaN-Episode_Length_squared2-pte_Length_per_Ads",
    "pte_Length_per_Ads-pte_Length_per_Host-pte_Expected_Listening_Time_Sentiment",
    "Guest_Popularity_percentage_NaN-Episode_Length_squared2-pte_Number_of_Ads",
    "Is_Positive_Sentiment-Expected_Listening_Time_Sentiment-pte_Length_per_Ads",
    "pte_Length_per_Ads-pte_Episode_Length_squared-pte_Expected_Listening_Time_Sentiment",
    "pte_Episode_Length_minutes-pte_Length_per_Ads-pte_Is_Positive_Sentiment",
    "ELen_Dec-Expected_Listening_Time_Sentiment-pte_Length_per_Ads",
    "Expected_Listening_Time_Sentiment-pte_Genre_Episode_Num_Length_per_Host_Is_Positive_Sentiment-pte_Length_per_Ads",
    "Expected_Listening_Time_Sentiment-pte_Length_per_Ads-pte_Length_per_Guest",
    "Expected_Listening_Time_Sentiment-pte_Length_per_Ads-pte_Length_per_Host",
    "Expected_Listening_Time_Sentiment-pte_Episode_Length_minutes_Number_of_Ads_HPperc_Int-pte_Length_per_Ads",
    "Expected_Listening_Time_Sentiment-pte_Genre_Host_Popularity_percentage_Episode_Num_ELen_Int-pte_Number_of_Ads",
    "Expected_Listening_Time_Sentiment-pte_Number_of_Ads-pte_Length_per_Guest",
    "Expected_Listening_Time_Sentiment-pte_Number_of_Ads-pte_Length_per_Host",
    "Publication_Time-Length_per_Ads-ELen_Int",
    "Episode_Sentiment-Expected_Listening_Time_Sentiment-pte_Number_of_Ads",
    "Episode_Sentiment-pte_Length_per_Ads-pte_Expected_Listening_Time_Sentiment",
]
if hasattr(cfg, "eval") and cfg.eval:
    before_fe_selecteds = random.sample(before_fe_selecteds, 20)
    selecteds = random.sample(selecteds, 20)


default_combinations_list = [
    # 2-interaction
    ["Episode_Length_minutes", "Host_Popularity_percentage"],
    ["Episode_Length_minutes", "Guest_Popularity_percentage"],
    ["Episode_Length_minutes", "Number_of_Ads"],
    ["Episode_Num", "Host_Popularity_percentage"],
    ["Episode_Num", "Guest_Popularity_percentage"],
    ["Episode_Num", "Number_of_Ads"],
    ["Host_Popularity_percentage", "Guest_Popularity_percentage"],
    ["Host_Popularity_percentage", "Number_of_Ads"],
    ["Host_Popularity_percentage", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Podcast_Name"],
    ["Episode_Num", "Podcast_Name"],
    ["Guest_Popularity_percentage", "Podcast_Name"],
    ["ELen_Int", "Episode_Num"],
    ["ELen_Int", "Host_Popularity_percentage"],
    ["ELen_Int", "Guest_Popularity_percentage"],
    ["ELen_Dec", "Episode_Num"],
    ["ELen_Dec", "Episode_Sentiment"],
    ["ELen_Dec", "Publication_Day"],
    # 3-interaction
    ["Episode_Length_minutes", "Episode_Num", "Host_Popularity_percentage"],
    ["Episode_Length_minutes", "Episode_Num", "Guest_Popularity_percentage"],
    ["Episode_Length_minutes", "Episode_Num", "Number_of_Ads"],
    ["Episode_Length_minutes", "Episode_Num", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Episode_Num", "Publication_Day"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Number_of_Ads"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Publication_Day"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Publication_Time"],
    ["Episode_Length_minutes", "Guest_Popularity_percentage", "Number_of_Ads"],
    ["Episode_Length_minutes", "Guest_Popularity_percentage", "Publication_Day"],
    ["Episode_Length_minutes", "Guest_Popularity_percentage", "Publication_Time"],
    ["Episode_Length_minutes", "Number_of_Ads", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Number_of_Ads", "Publication_Day"],
    ["Episode_Length_minutes", "Episode_Sentiment", "Publication_Time"],
    ["Episode_Num", "Host_Popularity_percentage", "Guest_Popularity_percentage"],
    ["Episode_Num", "Host_Popularity_percentage", "Number_of_Ads"],
    ["Episode_Num", "Host_Popularity_percentage", "Episode_Sentiment"],
    ["Episode_Num", "Host_Popularity_percentage", "Publication_Day"],
    ["Episode_Num", "Host_Popularity_percentage", "Publication_Time"],
    ["Episode_Num", "Host_Popularity_percentage", "Genre"],
    ["Episode_Num", "Guest_Popularity_percentage", "Number_of_Ads"],
    ["Episode_Num", "Guest_Popularity_percentage", "Episode_Sentiment"],
    ["Episode_Num", "Guest_Popularity_percentage", "Publication_Day"],
    ["Episode_Num", "Guest_Popularity_percentage", "Publication_Time"],
    ["Episode_Num", "Guest_Popularity_percentage", "Genre"],
    ["Episode_Num", "Number_of_Ads", "Episode_Sentiment"],
    ["Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"],
    ["Host_Popularity_percentage", "Guest_Popularity_percentage", "Episode_Sentiment"],
    ["Host_Popularity_percentage", "Guest_Popularity_percentage", "Publication_Day"],
    ["Host_Popularity_percentage", "Guest_Popularity_percentage", "Publication_Time"],
    ["Host_Popularity_percentage", "Number_of_Ads", "Publication_Day"],
    ["Guest_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment"],
    ["Guest_Popularity_percentage", "Number_of_Ads", "Genre"],
    ["ELen_Int", "Number_of_Ads", "Episode_Sentiment"],
    ["ELen_Dec", "Number_of_Ads", "Podcast_Name"],
    # 4-interaction
    ["Episode_Length_minutes", "Episode_Num", "Host_Popularity_percentage", "Guest_Popularity_percentage"],
    ["Episode_Length_minutes", "Episode_Num", "Host_Popularity_percentage", "Number_of_Ads"],
    ["Episode_Length_minutes", "Episode_Num", "Host_Popularity_percentage", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Episode_Num", "Host_Popularity_percentage", "Publication_Day"],
    ["Episode_Length_minutes", "Episode_Num", "Host_Popularity_percentage", "Publication_Time"],
    ["Episode_Length_minutes", "Episode_Num", "Host_Popularity_percentage", "Genre"],
    ["Episode_Length_minutes", "Episode_Num", "Guest_Popularity_percentage", "Number_of_Ads"],
    ["Episode_Length_minutes", "Episode_Num", "Guest_Popularity_percentage", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Episode_Num", "Guest_Popularity_percentage", "Publication_Day"],
    ["Episode_Length_minutes", "Episode_Num", "Guest_Popularity_percentage", "Publication_Time"],
    ["Episode_Length_minutes", "Episode_Num", "Number_of_Ads", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Episode_Num", "Number_of_Ads", "Publication_Day"],
    ["Episode_Length_minutes", "Episode_Num", "Number_of_Ads", "Publication_Time"],
    ["Episode_Length_minutes", "Episode_Num", "Publication_Day", "Publication_Time"],
    ["Episode_Length_minutes", "Episode_Num", "Publication_Day", "Genre"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Publication_Day"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Publication_Time"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Number_of_Ads", "Publication_Day"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Publication_Day", "Publication_Time"],
    ["Episode_Length_minutes", "Host_Popularity_percentage", "Publication_Day", "Genre"],
    ["Episode_Length_minutes", "Guest_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment"],
    ["Episode_Length_minutes", "Guest_Popularity_percentage", "Number_of_Ads", "Publication_Day"],
    ["Episode_Length_minutes", "Guest_Popularity_percentage", "Number_of_Ads", "Publication_Time"],
    ["Episode_Length_minutes", "Guest_Popularity_percentage", "Number_of_Ads", "Genre"],
    ["Episode_Length_minutes", "Episode_Num", "Publication_Time", "Podcast_Name"],
    ["Episode_Num", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"],
    ["Episode_Num", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Episode_Sentiment"],
    ["Episode_Num", "Host_Popularity_percentage", "Number_of_Ads", "Publication_Day"],
    ["Episode_Num", "Host_Popularity_percentage", "Number_of_Ads", "Publication_Time"],
    ["Episode_Num", "Host_Popularity_percentage", "Episode_Sentiment", "Publication_Day"],
    ["Episode_Num", "Host_Popularity_percentage", "Episode_Sentiment", "Publication_Time"],
    ["Episode_Num", "Host_Popularity_percentage", "Episode_Sentiment", "Genre"],
    ["Episode_Num", "Host_Popularity_percentage", "Publication_Day", "Publication_Time"],
    ["Episode_Num", "Host_Popularity_percentage", "Publication_Time", "Genre"],
    ["Episode_Num", "Guest_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment"],
    ["Episode_Num", "Guest_Popularity_percentage", "Number_of_Ads", "Genre"],
    ["Episode_Num", "Host_Popularity_percentage", "Episode_Sentiment", "Podcast_Name"],
    ["Host_Popularity_percentage", "Number_of_Ads", "Episode_Sentiment", "Podcast_Name"],
    ["Host_Popularity_percentage", "Number_of_Ads", "Publication_Day", "Podcast_Name"],
    ["Host_Popularity_percentage", "Number_of_Ads", "Publication_Time", "Podcast_Name"],
]
if hasattr(cfg, "eval") and cfg.eval:
    # default_combinations_list = [default_combinations_list[i] for i in range(0, len(default_combinations_list), 9)]
    # default_combinations_list = default_combinations_list[:20]
    default_combinations_list = random.sample(default_combinations_list, 20)

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

    e_median = df_train.select(pl.col("Episode_Length_minutes").median()).item()
    g_median = df_train.select(pl.col("Guest_Popularity_percentage").median()).item()
    n_median = df_train.select(pl.col("Number_of_Ads").median()).item()

    df = df.with_columns(
        pl.col("Episode_Length_minutes").fill_null(e_median),
        pl.col("Guest_Popularity_percentage").fill_null(g_median),
        pl.col("Number_of_Ads").fill_null(n_median),
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


def cols_encode(df: pl.DataFrame, combinations_list: list) -> pl.DataFrame:
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

    if hasattr(cfg, "default_combinations") and cfg.default_combinations:
        combinations_list = default_combinations_list.copy()
    else:
        combinations_list = [item.split("-") for item in before_fe_selecteds]

    print("Combinations list length:", len(combinations_list))

    X_train = cols_encode(X_train, combinations_list)
    X_valid = cols_encode(X_valid, combinations_list)
    if X_test is not None:
        X_test = cols_encode(X_test, combinations_list)

    encoded_columns = X_train.columns[before_encode_len:]

    datasetX = encode_target(y_train, encoded_columns, X_train, X_valid, X_test=X_test)
    X_train, X_valid, X_test = datasetX.get()

    encoded_columns = [col for col in encoded_columns if col != "Episode_Length_minutes"]
    datasetX = encode_target(X_train["Episode_Length_minutes"], encoded_columns, X_train, X_valid, X_test=X_test)
    X_train, X_valid, X_test = datasetX.get()

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

    # combinations_list = [[col] for col in numeric_cols] + default_combinations_list

    combinations_list = [item.split("-") for item in before_fe_selecteds]
    combinations_list += [[col] for col in numeric_cols]

    m = df_pltpd["Listening_Time_minutes"].mean()

    for cols in combinations_list:
        n = f"pte_{'_'.join(cols)}"
        means = df_pltpd.group_by(cols).agg(pl.col("Listening_Time_minutes").mean().alias("mean_listening_time"))
        df = df.join(means, on=cols, how="left").with_columns(pl.col("mean_listening_time").fill_null(m).alias(n)).drop("mean_listening_time")

    return df
