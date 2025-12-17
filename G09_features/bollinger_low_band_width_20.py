FEATURE_CODE = "bollinger_low_band_width_20"
import numpy as np, pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Bollinger Band Width (Low Price, 20-Period)
    Description:
    This feature measures the relative width of Bollinger Bands over a 20-period rolling window using the
    *low* price. It reflects short-term volatility of the lower price levels. Higher values indicate
    greater price dispersion and increased market volatility at the lows.

    Formula / Method:
    1. Compute 20-period rolling mean of low:
        m_t = rolling_mean(low_t, 20)
    2. Compute 20-period rolling standard deviation of low:
        sd_t = rolling_std(low_t, 20)
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
    m = g["low"].rolling(20, min_periods=20).mean()
    sd = g["low"].rolling(20, min_periods=20).std(ddof=0)
    s = (2 * sd) / m
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
