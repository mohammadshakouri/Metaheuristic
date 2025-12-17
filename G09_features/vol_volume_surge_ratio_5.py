FEATURE_CODE = "vol_volume_surge_ratio_5"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume Surge Ratio (5)
    Description:
        Compares the current traded volume to its recent 5-bar median to flag unusual surges or droughts.
    Formula / method (brief, cite if needed):
        volume / rolling_median(volume, window=5, min_periods=5); denominator zeros are treated as NaN.
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

    volume = g["volume"].astype(float)
    rolling_median = volume.rolling(window=5, min_periods=5).median()
    denominator = rolling_median.replace(0.0, np.nan)
    ratio = volume / denominator
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.astype(float)
    ratio.name = FEATURE_CODE
    return ratio
