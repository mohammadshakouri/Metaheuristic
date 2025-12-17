FEATURE_CODE = "geo_struct_regime_score_50"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Structural Regime Score over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close = g["close"]
    regime_score = pd.Series(index=close.index, dtype=float)

    for i in range(49, len(close)):
        y = close[i-49:i+1]
        x = np.arange(50)
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        regime_score.iat[i] = m

    regime_score.name = FEATURE_CODE
    return regime_score
