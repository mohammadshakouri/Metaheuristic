FEATURE_CODE = "hybrid_vol_mom_regime_score_50"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Volatility-Momentum Regime Score over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    momentum = g["close"].diff(10)

    regime_score = (momentum / volatility.replace(0, np.nan)).rolling(50, min_periods=50).mean()

    s = regime_score.astype(float)
    s.name = FEATURE_CODE
    return s
