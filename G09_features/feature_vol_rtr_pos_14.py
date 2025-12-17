FEATURE_CODE = "vol_rtr_pos_14"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Relative True Range Position (14)

    Description:
    Measures where the current True Range (TR) falls within the
    last 14 bars' TR min-max range. Values:
    0 = lowest volatility in window
    1 = highest volatility in window
    0.5 = mid-range

    Formula:
    RTR_t = (TR_t - min(TR_{t-13:t})) / (max(TR_{t-13:t}) - min(TR_{t-13:t}))

    Input:
    OHLCV DataFrame

    Output:
    pd.Series(float), name == FEATURE_CODE.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high, low, close = g["high"], g["low"], g["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    tr_min = tr.rolling(14, min_periods=14).min()
    tr_max = tr.rolling(14, min_periods=14).max()

    s = (tr - tr_min) / (tr_max - tr_min).replace(0, np.nan)
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
