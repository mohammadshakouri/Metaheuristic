FEATURE_CODE = "trend_cum_return"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Cumulative normalized return

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    r = g["close"].pct_change().fillna(0)
    s = (1 + r).cumprod() - 1
    s = s.astype(float); s.name = FEATURE_CODE
    return s
