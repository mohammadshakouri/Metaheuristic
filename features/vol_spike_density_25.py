FEATURE_CODE = "vol_spike_density_25"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volatility Spike Density over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    spike_threshold = volatility.rolling(25, min_periods=25).mean() + 2 * volatility.rolling(25, min_periods=25).std()
    spikes = (volatility > spike_threshold).astype(int)
    vsd = spikes.rolling(25, min_periods=25).mean()

    s = vsd.astype(float)
    s.name = FEATURE_CODE
    return s
