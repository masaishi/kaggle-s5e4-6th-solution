from dataclasses import dataclass
from pathlib import Path

import altair as alt
import polars as pl

alt.renderers.enable("jupyter", offline=True)
alt.data_transformers.disable_max_rows()

import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


@dataclass
class CFG:
    train_path: Path = Path("/kaggle/input/playground-series-s5e4/train.csv")
    test_path: Path = Path("/kaggle/input/playground-series-s5e4/test.csv")
    sub_path: Path = Path("./data/sample_submission.csv")


cfg = CFG()


df_train = pl.read_csv(cfg.train_path)
df_train = df_train.filter(pl.col("Number_of_Ads").is_not_null())

df_train = df_train.drop("id")


df = df_train.clone()
df
