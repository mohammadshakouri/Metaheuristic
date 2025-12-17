FEATURE_CODE = "info_range_entropy_20"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Entropy of normalized range (high-low)/close

    Automatically generated for Phase 1 — Metaheuristic Course.
    """
    g = df.copy(); g.columns = [str(c).lower() for c in g.columns]
    r = ((g["high"] - g["low"]) / g["close"]).replace([np.inf, -np.inf], np.nan)
    def ent(x):
        bins = np.histogram(x, bins=5, density=True)[0]
        p = bins[bins > 0]
        return -np.sum(p * np.log2(p))
    s = r.rolling(20, min_periods=20).apply(ent, raw=True)
    s = s.astype(float); s.name = FEATURE_CODE
    return s
