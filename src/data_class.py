from dataclasses import dataclass
from typing import Optional

import polars as pl


@dataclass
class DatasetX:
    X_train: pl.DataFrame
    X_valid: pl.DataFrame
    X_test: Optional[pl.DataFrame] = None

    def get(self) -> pl.DataFrame:
        return self.X_train, self.X_valid, self.X_test

    def to_dataset_xy(self, y_train: pl.Series, y_valid: pl.Series, y_test: Optional[pl.Series] = None) -> "DatasetXy":
        return DatasetXy(X_train=self.X_train, y_train=y_train, X_valid=self.X_valid, y_valid=y_valid, X_test=self.X_test, y_test=y_test)


@dataclass
class DatasetXy:
    X_train: pl.DataFrame
    y_train: pl.Series
    X_valid: pl.DataFrame
    y_valid: pl.Series
    X_test: Optional[pl.DataFrame] = None
    y_test: Optional[pl.Series] = None

    def get(self) -> pl.DataFrame:
        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test

    def to_dataset_x(self) -> DatasetX:
        return DatasetX(X_train=self.X_train, X_valid=self.X_valid, X_test=self.X_test)


@dataclass
class Dfs:
    df_train: pl.DataFrame
    df_valid: pl.DataFrame
    df_test: Optional[pl.DataFrame] = None

    def get(self) -> pl.DataFrame:
        return self.df_train, self.df_valid, self.df_test
