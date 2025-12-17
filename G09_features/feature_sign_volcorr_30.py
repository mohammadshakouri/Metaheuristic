FEATURE_CODE = "sign_volcorr_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Rolling correlation between volume and return sign

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    r = np.sign(g["close"].pct_change().fillna(0))
    s = g["volume"].rolling(30, min_periods=30).corr(r)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
