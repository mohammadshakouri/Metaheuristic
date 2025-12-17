FEATURE_CODE = "liq_spike_persistence_15"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Spike Persistence over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    vol_changes = g["volume"].pct_change()
    spike_threshold = vol_changes.rolling(15, min_periods=15).mean() + 2 * vol_changes.rolling(15, min_periods=15).std()
    spikes = (vol_changes > spike_threshold).astype(int)
    lsp = spikes.rolling(15, min_periods=15).mean()

    s = lsp.astype(float)
    s.name = FEATURE_CODE
    return s
