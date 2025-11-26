FEATURE_CODE = "geo_price_reversal_flag_15"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Price Reversal Flag over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    reversal_flag = delta.rolling(15, min_periods=15).apply(
        lambda x: 1 if (x[-1] * x[0] < 0) else 0
    )

    s = reversal_flag.astype(float)
    s.name = FEATURE_CODE
    return s
