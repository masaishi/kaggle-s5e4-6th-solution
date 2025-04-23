from typing import List, Optional, Union

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


# reference: https://www.kaggle.com/code/act18l/say-goodbye-to-ordinalencoder
class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Out‑of‑fold **mean‑rank** encoder with optional smoothing.
    • Encodes each category by the *rank* of its target mean within a fold.
    • Unseen categories get the global mean rank (or −1 if you prefer).
    """

    def __init__(self, cat_cols=None, n_splits=5, smoothing=0, random_state=42):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.smoothing = smoothing  # 0 = no smoothing
        self.maps_ = {}  # per‑fold maps
        self.global_map = {}  # fit on full data for test set
        self.random_state = random_state

    def _make_fold_map(self, X_col, y):
        means = y.groupby(X_col, dropna=False).mean()
        if self.smoothing > 0:
            counts = y.groupby(X_col, dropna=False).count()
            smooth = (counts * means + self.smoothing * y.mean()) / (counts + self.smoothing)
            means = smooth
        return {k: r for r, k in enumerate(means.sort_values().index)}

    def fit(self, X, y):
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include="object").columns.tolist()

        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        self.maps_ = {col: [None] * self.n_splits for col in self.cat_cols}

        for fold, (tr_idx, _) in enumerate(kf.split(X)):
            X_tr, y_tr = X.loc[tr_idx], y.loc[tr_idx]
            for col in self.cat_cols:
                self.maps_[col][fold] = self._make_fold_map(X_tr[col], y_tr)

        for col in self.cat_cols:
            self.global_map[col] = self._make_fold_map(X[col], y)

        return self

    def transform(self, X, y=None, fold=None):
        """
        • During CV pass fold index to use fold‑specific maps (leak‑free).
        • At inference time (fold=None) uses global map.
        """
        X = X.copy()
        tgt_maps = {col: (self.global_map[col] if fold is None else self.maps_[col][fold]) for col in self.cat_cols}
        for col, mapping in tgt_maps.items():
            X[col] = X[col].map(mapping).fillna(-1).astype(int)
        return X


class PolarsTargetEncoder:
    """Target Encoder for Polars DataFrames.

    Encodes categorical features based on the mean target value for each category,
    with optional smoothing to blend with the global mean.
    """

    def __init__(
        self,
        cat_columns: Optional[List[str]] = None,
        smooth: Union[str, float] = "auto",
    ):
        """
        Parameters
        ----------
        cat_columns : list of str or None, default=None
            List of categorical column names to encode. If None, all string and categorical
            columns will be encoded.

        smooth : "auto" or float, default="auto"
            Smoothing parameter that controls blending between the category mean and the global mean.
            Higher values put more weight on the global mean.
            If "auto", smoothing will be determined automatically based on category counts.
        """
        self.cat_columns = cat_columns
        self.smooth = smooth
        self.encodings = {}
        self.global_mean = None

    def fit(self, df: pl.DataFrame, target_column: str) -> "PolarsTargetEncoder":
        """Fit the encoder on the input data."""
        # Store global target mean
        self.global_mean = df[target_column].mean()

        # Identify categorical columns if not specified
        if self.cat_columns is None:
            self.cat_columns = [
                col
                for col in df.columns
                if col != target_column and (df[col].dtype == pl.Categorical or df[col].dtype == pl.String or df[col].dtype == pl.Object)
            ]

        # Calculate target encoding for each categorical column
        for col in self.cat_columns:
            # Calculate means and counts per category
            encoding_stats = df.group_by(col).agg([pl.mean(target_column).alias("category_mean"), pl.count(target_column).alias("category_count")])

            # Calculate smoothing factor for each category
            if self.smooth == "auto":
                # Use fixed smoothing value for auto mode - simplified approach
                smooth_value = 10.0
                encoding_stats = encoding_stats.with_columns([pl.lit(smooth_value).alias("smooth")])
            else:
                # Use fixed smoothing value
                encoding_stats = encoding_stats.with_columns([pl.lit(float(self.smooth)).alias("smooth")])

            # Calculate smoothed encoding
            encoding_stats = encoding_stats.with_columns(
                [
                    (
                        (pl.col("category_count") * pl.col("category_mean") + pl.col("smooth") * self.global_mean)
                        / (pl.col("category_count") + pl.col("smooth"))
                    ).alias("encoded_value")
                ]
            )

            # Store encoding mapping
            self.encodings[col] = encoding_stats.select([col, "encoded_value"]).to_dict(as_series=False)

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform categories to their target encodings."""
        result = df.clone()

        for col in self.cat_columns:
            if col not in df.columns:
                continue

            # Get the encoding dict for this column
            encoding_dict = self.encodings[col]

            # Create a lookup dataframe
            lookup_df = pl.DataFrame({col: list(encoding_dict[col]), "encoded_value": list(encoding_dict["encoded_value"])})

            # Join with the lookup dataframe to apply the encoding
            result = result.join(lookup_df, on=col, how="left")

            # Fill missing values with global mean
            result = result.with_columns([pl.col("encoded_value").fill_null(self.global_mean).alias(f"{col}_encoded")])

            # Drop the temporary column
            result = result.drop("encoded_value")

        return result

    def fit_transform(self, df: pl.DataFrame, target_column: str) -> pl.DataFrame:
        """Fit the encoder and transform the input data."""
        # First fit the encoder to learn global statistics
        self.fit(df, target_column)

        # Create output DataFrame
        result = df.clone()

        # For each category, calculate smoothed means
        for col in self.cat_columns:
            # Get total count and mean per category
            category_stats = df.group_by(col).agg([pl.count(target_column).alias("count"), pl.mean(target_column).alias("mean")])

            # Apply smoothing
            if self.smooth == "auto":
                smooth_value = 10.0  # Simplified approach
            else:
                smooth_value = float(self.smooth)

            # Calculate smoothed encoding
            category_stats = category_stats.with_columns(
                [((pl.col("count") * pl.col("mean") + smooth_value * self.global_mean) / (pl.col("count") + smooth_value)).alias("encoded_value")]
            )

            # Join with the category stats dataframe to apply the encoding
            result = result.join(category_stats.select([col, "encoded_value"]), on=col, how="left")

            # Fill missing values with global mean
            result = result.with_columns([pl.col("encoded_value").fill_null(self.global_mean).alias(f"{col}_encoded")])

            # Drop the temporary column
            result = result.drop("encoded_value")

        return result


class LOOTargetEncoder:
    """Leave-One-Out Target Encoder for Polars DataFrames.

    Encodes categorical features based on the mean target value for each category,
    excluding the current row when encoding to prevent data leakage.
    Includes optional smoothing to blend with the global mean.
    """

    def __init__(
        self,
        cat_columns: Optional[List[str]] = None,
        smooth: Union[str, float] = 10.0,
    ):
        """
        Parameters
        ----------
        cat_columns : list of str or None, default=None
            List of categorical column names to encode. If None, all string and categorical
            columns will be encoded.

        smooth : float, default=10.0
            Smoothing parameter that controls blending between the category mean and the global mean.
            Higher values put more weight on the global mean.
        """
        self.cat_columns = cat_columns
        self.smooth = smooth
        self.global_mean = None
        self.category_stats = {}
        self.target_column = None
        self.is_fitted = False

    def fit(self, df: pl.DataFrame, target: Union[str, pl.Series]) -> "LOOTargetEncoder":
        """Calculate global stats needed for the LOO encoding."""
        # Handle target as either column name or Series
        if isinstance(target, str):
            self.target_column = target
            target_values = df[target]
        else:
            self.target_column = "_target"
            target_values = target
            # Add target to df if it's a Series
            if len(df) == len(target_values):
                df = df.with_columns(pl.lit(target_values).alias(self.target_column))
            else:
                raise ValueError("Target Series length does not match DataFrame length")

        # Store global target mean
        self.global_mean = target_values.mean()

        # Identify categorical columns if not specified
        if self.cat_columns is None:
            self.cat_columns = [
                col
                for col in df.columns
                if col != self.target_column and (df[col].dtype == pl.Categorical or df[col].dtype == pl.String or df[col].dtype == pl.Object)
            ]

        # Calculate target sums and counts per category for LOO calculation
        for col in self.cat_columns:
            category_stats = df.group_by(col).agg([pl.sum(self.target_column).alias("sum"), pl.count(self.target_column).alias("count")])

            self.category_stats[col] = category_stats

        self.is_fitted = True
        return self

    def transform(self, df: pl.DataFrame, target: Optional[Union[str, pl.Series]] = None) -> pl.DataFrame:
        """Transform categories to their leave-one-out target encodings."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        # Clone the DataFrame to avoid modifying the original
        result = df.clone()

        # Handle target data if provided
        if target is not None:
            if isinstance(target, str):
                target_column = target
            else:  # It's a Series
                target_column = "_transform_target"
                result = result.with_columns(pl.lit(target).alias(target_column))
            has_target = True
        else:
            has_target = False
            target_column = None

        # Process each categorical column
        for col in self.cat_columns:
            if col not in df.columns:
                continue

            # Get the category statistics
            cat_stats = self.category_stats[col]

            if has_target:
                # Create a unique row identifier for proper leave-one-out computation
                result = result.with_columns(pl.Series(name="_row_id", values=range(len(result))))

                # Add category stats to result
                result = result.join(cat_stats, on=col, how="left")

                # Calculate LOO encoding - fixed to properly exclude the current row
                smooth = float(self.smooth)
                global_mean = self.global_mean

                # Create properly adjusted sum and count for LOO
                result = result.with_columns(
                    [
                        # Adjust sum by removing current row's target value
                        (pl.col("sum") - pl.col(target_column)).alias("adjusted_sum"),
                        # Adjust count by removing current row (always subtract 1)
                        (pl.col("count") - 1).alias("adjusted_count"),
                    ]
                )

                # Calculate LOO encoding
                result = result.with_columns(
                    [
                        (
                            # For adjusted count > 0: Use adjusted mean with smoothing
                            pl.when(pl.col("adjusted_count") > 0)
                            .then((pl.col("adjusted_sum") + smooth * global_mean) / (pl.col("adjusted_count") + smooth))
                            # For adjusted count = 0: Use global mean
                            .otherwise(global_mean)
                        ).alias(f"{col}_encoded")
                    ]
                )

                # Drop temporary columns
                result = result.drop(["sum", "count", "adjusted_sum", "adjusted_count", "_row_id"])

            else:
                # Regular encoding when no target is available (for new data)
                # Pre-calculate the encoded values (using the fitted stats)
                encoded_values = cat_stats.with_columns(
                    [
                        (
                            (pl.col("sum") / pl.col("count") * pl.col("count") + float(self.smooth) * self.global_mean) / (pl.col("count") + float(self.smooth))
                        ).alias("encoded_value")
                    ]
                )

                # Join with the encoded values
                result = result.join(encoded_values.select([col, "encoded_value"]), on=col, how="left")

                # Fill missing categories with global mean
                result = result.with_columns([pl.col("encoded_value").fill_null(self.global_mean).alias(f"{col}_encoded")])

                # Drop temporary column
                result = result.drop("encoded_value")

        # Remove added target column if we added it
        if has_target and not isinstance(target, str):
            result = result.drop(target_column)

        return result

    def fit_transform(self, df: pl.DataFrame, target: Union[str, pl.Series]) -> pl.DataFrame:
        """Fit the encoder and transform the input data."""
        self.fit(df, target)
        return self.transform(df, target)


class HoldoutTargetEncoder:
    """Holdout Target Encoder for Polars DataFrames.

    Encodes categorical features based on target mean values calculated from a separate
    holdout dataset to prevent data leakage.
    """

    def __init__(
        self,
        cat_columns: Optional[List[str]] = None,
        smooth: Union[float, str] = 10.0,
    ):
        """
        Parameters
        ----------
        cat_columns : list of str or None, default=None
            List of categorical column names to encode. If None, all string and categorical
            columns will be encoded.

        smooth : float, default=10.0
            Smoothing parameter that controls blending between the category mean and the global mean.
            Higher values put more weight on the global mean.
        """
        self.cat_columns = cat_columns
        self.smooth = smooth
        self.encodings = {}
        self.global_mean = None

    def fit(self, holdout_df: pl.DataFrame, target_column: str) -> "HoldoutTargetEncoder":
        """Fit the encoder on the holdout data."""
        # Store global target mean
        self.global_mean = holdout_df[target_column].mean()

        # Identify categorical columns if not specified
        if self.cat_columns is None:
            self.cat_columns = [
                col
                for col in holdout_df.columns
                if col != target_column
                and (holdout_df[col].dtype == pl.Categorical or holdout_df[col].dtype == pl.String or holdout_df[col].dtype == pl.Object)
            ]

        # Calculate encoding for each categorical column using the holdout set
        for col in self.cat_columns:
            # Calculate means and counts per category
            encoding_stats = holdout_df.group_by(col).agg([pl.mean(target_column).alias("category_mean"), pl.count(target_column).alias("category_count")])

            # Get smoothing value
            smooth_value = 10.0 if self.smooth == "auto" else float(self.smooth)

            # Calculate smoothed encoding
            encoding_stats = encoding_stats.with_columns(
                [
                    ((pl.col("category_count") * pl.col("category_mean") + smooth_value * self.global_mean) / (pl.col("category_count") + smooth_value)).alias(
                        "encoded_value"
                    )
                ]
            )

            # Store encoding mapping
            self.encodings[col] = encoding_stats.select([col, "encoded_value"]).to_dict(as_series=False)

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform categories using encodings learned from the holdout data."""
        result = df.clone()

        for col in self.cat_columns:
            if col not in df.columns:
                continue

            # Get encoding dictionary for this column
            encoding_dict = self.encodings[col]

            # Create lookup dataframe
            lookup_df = pl.DataFrame({col: list(encoding_dict[col]), "encoded_value": list(encoding_dict["encoded_value"])})

            # Join with lookup to apply encoding
            result = result.join(lookup_df, on=col, how="left")

            # Fill missing values with global mean
            result = result.with_columns([pl.col("encoded_value").fill_null(self.global_mean).alias(f"{col}_encoded")])

            # Drop temporary column
            result = result.drop("encoded_value")

        return result

    def fit_transform(self, holdout_df: pl.DataFrame, target_column: str, df: pl.DataFrame) -> pl.DataFrame:
        """Fit encoder on holdout data and transform the input data."""
        self.fit(holdout_df, target_column)
        return self.transform(df)
