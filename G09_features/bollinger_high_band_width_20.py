FEATURE_CODE = "bollinger_high_band_width_20"
import numpy as np, pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Bollinger Band Width (High Price, 20-Period)
    Description:
    This feature measures the relative width of Bollinger Bands over a 20-period rolling window using the
    *high* price. It captures short-term volatility of the upper price levels. Higher values indicate
    larger price dispersion and more volatile highs.

    Formula / Method:
    1. Compute 20-period rolling mean of high:
        m_t = rolling_mean(high_t, 20)
    2. Compute 20-period rolling standard deviation of high:
        sd_t = rolling_std(high_t, 20)
    3. Calculate Bollinger Band width:
        BB_width_t = (2 * sd_t) / m_t

    Input:
    df: DataFrame with DatetimeIndex (ascending), columns:
    open, high, low, close, volume (case-insensitive)

    Output:
    pd.Series (float), same index as df.index, length == len(df),
    name == FEATURE_CODE. Initial 19 values will be NaN due to rolling window.

    Constraints:
    - No look-ahead (only uses current and past data).
    - Fully vectorized using pandas/numpy (no Python loops).
    - NaNs at the start are expected and acceptable.
    """

    g = df.rename(columns=str.lower).astype(float)
    m = g["high"].rolling(20, min_periods=20).mean()
    sd = g["high"].rolling(20, min_periods=20).std(ddof=0)
    s = (2 * sd) / m
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
