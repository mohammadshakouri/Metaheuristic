FEATURE_CODE = "vol_cluster_density_40"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Volatility Cluster Density over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    mean_vol = volatility.rolling(40, min_periods=40).mean()

    cluster_density = (volatility > mean_vol).rolling(40, min_periods=40).mean()

    s = cluster_density.astype(float)
    s.name = FEATURE_CODE
    return s
