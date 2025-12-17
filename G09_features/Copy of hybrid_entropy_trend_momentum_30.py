FEATURE_CODE = "hybrid_entropy_trend_momentum_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Entropy-Trend-Momentum Indicator over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    trend = delta.rolling(30, min_periods=30).mean()
    momentum = delta.rolling(30, min_periods=30).std()

    entropy = -trend.rolling(30, min_periods=30).apply(
        lambda x: np.sum((x / x.sum()) * np.log((x / x.sum()) + 1e-10))
    )

    htm = (trend * momentum * entropy).rolling(30, min_periods=30).mean()

    s = htm.astype(float)
    s.name = FEATURE_CODE
    return s
