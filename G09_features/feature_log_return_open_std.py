FEATURE_CODE = "log_return_open_std"
import numpy as np, pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Volatility of Log Returns (Open Price, 20-Period)
    Description:
    This feature measures the volatility of the day-to-day logarithmic changes in the *open* price over a
    20-day rolling window. It captures how stable or unstable the market’s opening prices have been recently.
    Higher values indicate more uncertainty or turbulence in how the market opens day to day.

    Formula / Method:
    1. Compute daily log return of open price:
        logret_t = ln(open_t / open_(t-1))
    2. Compute 20-period rolling standard deviation:
        volatility_t = std(logret_(t-19 : t)), population std (ddof=0)

    Input:
    df: DataFrame with DatetimeIndex (ascending), columns:
    open, high, low, close, volume (case-insensitive)

    Output:
    pd.Series (float), same index as df.index, length == len(df),
    name == FEATURE_CODE. The first 19 values will be NaN because a full 20-period window is required.

    Constraints:
    - No look-ahead (uses only current and past data).
    - Fully vectorized using pandas/numpy (no Python loops).
    - NaNs at the beginning are expected and acceptable.
    """

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]
    logret = np.log(g["open"] / g["open"].shift(1))
    s = logret.rolling(20, min_periods=20).std(ddof=0)
    s.name = FEATURE_CODE
    return s
