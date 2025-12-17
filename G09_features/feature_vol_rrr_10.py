FEATURE_CODE = "vol_rrr_10"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Rolling Range Ratio (10)

    Description:
    Measures the current candle’s high–low range relative to the
    average range of the last 10 bars. High values indicate expansion.

    Formula:
    range_t = high - low
    rrr_t = range_t / SMA(range,10)

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    rng = g["high"] - g["low"]
    avg = rng.rolling(10, min_periods=10).mean()

    s = rng / avg.replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
