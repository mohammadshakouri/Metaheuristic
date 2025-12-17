FEATURE_CODE = "entropy_hour_agg"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Entropy of returns aggregated by hour of day

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    r = g["close"].pct_change()
    grouped = r.groupby(g.index.hour).mean()
    p = (grouped - grouped.min()) / (grouped.max() - grouped.min() + 1e-9)
    p = p[p > 0]
    ent = -np.sum(p * np.log2(p))
    s = pd.Series(ent, index=g.index)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
