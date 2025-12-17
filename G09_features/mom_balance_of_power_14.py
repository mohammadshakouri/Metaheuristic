FEATURE_CODE = "mom_balance_of_power_14"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Balance of Power (14)
    Description:
        Tracks the average balance between bullish and bearish pressure by smoothing the Balance of Power measure.
    Formula / method (brief, cite if needed):
        Balance of Power = (close - open) / (high - low); feature is a 14-bar rolling mean of this ratio (denominator zeros ignored).
    Input:
        df: DataFrame with DatetimeIndex (ascending), columns:
        open, high, low, close, volume (case-insensitive)
    Output:
        pd.Series (float preferred), same index as df.index, length == len(df),
        name == FEATURE_CODE. Initial NaNs from rolling windows are OK.
    Constraints:
        - No look-ahead (use only current and past data at each row).
        - Vectorized (avoid Python loops when possible).
        - Use only numpy and pandas.
    """
    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    open_ = g["open"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)

    range_ = high - low
    range_ = range_.replace(0.0, np.nan)

    bop_raw = (close - open_) / range_
    bop = bop_raw.rolling(window=14, min_periods=14).mean()
    bop = bop.astype(float)
    bop.name = FEATURE_CODE
    return bop
