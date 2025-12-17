FEATURE_CODE = "vol_mad_ratio_30"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Median Absolute Deviation Ratio (30)

    Description:
    Measures how far the closing price deviates from the rolling
    30-period median, normalized by the Median Absolute Deviation (MAD).
    Robust to outliers compared to standard deviation. High values
    indicate unusually large deviations in price.

    Formula:
    med_t = median(close_{t-29:t})
    MAD_t = median(|close - med_t|)
    mad_ratio_t = (close_t - med_t) / MAD_t

    Input:
    df: DataFrame with open, high, low, close, volume (case-insensitive)

    Output:
    pd.Series (float), same index and length as df, name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]
    close = g["close"]

    med = close.rolling(30, min_periods=30).median()
    mad = (close - med).abs().rolling(30, min_periods=30).median()

    s = (close - med) / mad.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
