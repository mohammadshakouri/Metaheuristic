FEATURE_CODE = "vol_vmo_14"

import numpy as np
import pandas as pd


def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volume Momentum Oscillator (14)

    Description:
    Uses fast and slow EMAs of volume to detect volume surges
    or contractions.

    Formula:
    fast = EMA(volume,6)
    slow = EMA(volume,14)
    vmo = fast - slow

    Output: pd.Series(float)
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    vol = g["volume"].astype(float)
    fast = vol.ewm(span=6, adjust=False).mean()
    slow = vol.ewm(span=14, adjust=False).mean()

    s = fast - slow
    s = s.astype(float)
    s.name = FEATURE_CODE
    return s
