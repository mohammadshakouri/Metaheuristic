FEATURE_CODE = "info_cross_entropy_price_volume_30"
import numpy as np, pandas as pd

def compute_feature(df: pd.DataFrame) -> pd.Series:
    """
    Information Cross Entropy between Price and Volume over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_returns = g["close"].pct_change()
    vol_changes = g["volume"].pct_change()

    cross_entropy = pd.Series(index=g.index, dtype=float)

    for i in range(29, len(g)):
        pr = price_returns[i-29:i+1]
        vc = vol_changes[i-29:i+1]
        joint_dist = pd.crosstab(pr, vc, normalize=True)
        marginal_pr = pr.value_counts(normalize=True)
        ce = -np.sum(joint_dist.values * np.log(marginal_pr.reindex(joint_dist.index, fill_value=1e-10).values + 1e-10))
        cross_entropy.iat[i] = ce

    cross_entropy.name = FEATURE_CODE
    return cross_entropy
