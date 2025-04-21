from typing import List, Optional, Union

import polars as pl


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
