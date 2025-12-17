FEATURE_CODE = "entropy_return_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """Shannon entropy of return bins (30)."""

    g = df.copy()
    g.columns = [str(c).lower() for c in g.columns]

    r = g["close"].pct_change()

    def ent_window(x: pd.Series):
        bins = np.histogram(x, bins=5, density=True)[0]
        p = bins[bins > 0]
        return -np.sum(p * np.log2(p))

    s = r.rolling(30, min_periods=30).apply(ent_window, raw=False)

    s = s.astype(float)
    s.name = FEATURE_CODE
    return s.reindex(df.index)
