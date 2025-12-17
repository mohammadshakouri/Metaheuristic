FEATURE_CODE = "log_return_low_std"
import numpy as np, pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Volatility of Log Returns (low Price, 20-Period)
    Description:
    This feature measures the volatility of the day-to-day logarithmic changes in the *low* price over the
    most recent 20 periods. It reflects how much the upper price extremes fluctuate over time. lower values
    indicate more instability or expansion in price range, while higher values indicate calmer market lows.

    Formula / Method:
    1. Compute daily log return of high price:
        logret_t = ln(low_t / low_(t-1))
    2. Compute 20-period rolling standard deviation (population std, ddof=0):
        volatility_t = std(logret_(t-19 : t))

    Input:
    df: DataFrame with DatetimeIndex (ascending), columns:
    open, high, low, close, volume (case-insensitive)

    Output:
    pd.Series (float), same index as df.index, length == len(df),
    name == FEATURE_CODE. The first 19 values will be NaN due to the rolling window requirement.

    Constraints:
    - No look-ahead bias (each value uses only current and past observations).
    - Fully vectorized using pandas and numpy (no Python loops).
    - Initial NaNs are expected and acceptable.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    logret = np.log(g["low"] / g["low"].shift(1))
    s = logret.rolling(20, min_periods=20).std(ddof=0)
    s.name = FEATURE_CODE
    return s
