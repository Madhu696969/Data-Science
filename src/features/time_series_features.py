import pandas as pd
import numpy as np


def create_time_series_features(data):

    df = data.copy()

    # 1. Sort chronologically
    df = df.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    # 2. Group sales by store and item
    grouped = df.groupby(
        ["store", "item"],
        sort=False
    )["sales"]

    # 3. Lag features
    lags = [1, 7, 14, 28, 30]

    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # 4. Rolling mean and standard deviation
    windows = [7, 14, 30]

    for window in windows:

        df[f"rolling_mean_{window}"] = (
            grouped.transform(
                lambda x: x.shift(1)
                .rolling(window)
                .mean()
            )
        )

        df[f"rolling_std_{window}"] = (
            grouped.transform(
                lambda x: x.shift(1)
                .rolling(window)
                .std()
            )
        )

    # 5. Expanding statistics
    df["expanding_mean"] = (
        grouped.transform(
            lambda x: x.shift(1)
            .expanding()
            .mean()
        )
    )

    df["expanding_std"] = (
        grouped.transform(
            lambda x: x.shift(1)
            .expanding()
            .std()
        )
    )

    # 6. Date features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )

    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_month_start"] = (
        df["date"].dt.is_month_start.astype(int)
    )

    df["is_month_end"] = (
        df["date"].dt.is_month_end.astype(int)
    )

    # 7. Weekly cyclical encoding
    df["week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # 8. Yearly cyclical encoding
    df["year_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["year_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    return df