FEATURE_CODE = "mom_entropy_weighted_roc_15"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Entropy-Weighted Rate of Change over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].pct_change(15)
    entropy = -price_change.rolling(15, min_periods=15).apply(
        lambda x: np.sum((x + 1) * np.log(x + 1 + 1e-10)) / np.log(len(x))
    )

    ewr = price_change * entropy

    s = ewr.astype(float)
    s.name = FEATURE_CODE
    return s
