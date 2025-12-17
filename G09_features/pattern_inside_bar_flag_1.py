FEATURE_CODE = "pattern_inside_bar_flag_1"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Inside Bar Flag (1)
    Description:
        Identifies when the current bar is fully contained within the previous bar's range, signaling potential compression.
    Formula / method (brief, cite if needed):
        Flag = 1 if high_t <= high_{t-1} and low_t >= low_{t-1}; otherwise 0. First bar yields NaN.
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

    high = g["high"].astype(float)
    low = g["low"].astype(float)

    inside = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(float)
    inside.name = FEATURE_CODE
    inside.iloc[0] = np.nan
    return inside
