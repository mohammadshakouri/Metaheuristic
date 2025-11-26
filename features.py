import numpy as np, pandas as pd

def dyn_trend_slope_ema_20(df: pd.DataFrame) -> pd.Series:
    """
    Slope of EMA(20) as trend speed indicator.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    ema20 = g["close"].ewm(span=20, adjust=False).mean()
    slope = ema20.diff()  # first difference as slope proxy

    s = slope.astype(float)
    s.name = "dyn_trend_slope_ema_20"
    return s

def dyn_trend_strength_ratio_50(df: pd.DataFrame) -> pd.Series:
    """
    Trend Strength Ratio over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    up = delta.clip(lower=0).rolling(50, min_periods=50).sum()
    down = -delta.clip(upper=0).rolling(50, min_periods=50).sum()

    tsr = up / down.replace(0, np.nan)

    s = tsr.astype(float)
    s.name = "dyn_trend_strength_ratio_50"
    return s

def dyn_trend_angle_30(df: pd.DataFrame) -> pd.Series:
    """
    Trend Angle based on 30-period linear regression.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close = g["close"]
    angles = pd.Series(index=close.index, dtype=float)

    for i in range(29, len(close)):
        y = close[i-29:i+1]
        x = np.arange(30)
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        angle = np.arctan(m) * (180 / np.pi)
        angles.iat[i] = angle

    angles.name = "dyn_trend_angle_30"
    return angles

def dyn_trend_reversal_index_40(df: pd.DataFrame) -> pd.Series:
    """
    Trend Reversal Index over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    rev_up = delta.clip(lower=0).rolling(40, min_periods=40).sum()
    rev_down = -delta.clip(upper=0).rolling(40, min_periods=40).sum()

    tri = rev_down / rev_up.replace(0, np.nan)

    s = tri.astype(float)
    s.name = "dyn_trend_reversal_index_40"
    return s

def dyn_trend_consistency_25(df: pd.DataFrame) -> pd.Series:
    """
    Trend Consistency over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    pos_days = (delta > 0).rolling(25, min_periods=25).sum()
    neg_days = (delta < 0).rolling(25, min_periods=25).sum()

    tc = pos_days / (pos_days + neg_days).replace(0, np.nan)

    s = tc.astype(float)
    s.name = "dyn_trend_consistency_25"
    return s

def mom_impulse_strength_10(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Impulse Strength over 10 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    momentum = g["close"].diff(10)
    volatility = g["close"].rolling(10, min_periods=10).std()

    mis = momentum / volatility.replace(0, np.nan)

    s = mis.astype(float)
    s.name = "mom_impulse_strength_10"
    return s

def mom_energy_osc_20(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Energy Oscillator over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    momentum = g["close"].diff(1)
    energy = momentum.rolling(20, min_periods=20).apply(lambda x: np.sum(x**2))

    meo = momentum / energy.replace(0, np.nan)

    s = meo.astype(float)
    s.name = "mom_energy_osc_20"
    return s

def mom_vol_weighted_roc_15(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Rate of Change over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].pct_change(15)
    vol_weight = g["volume"] / g["volume"].rolling(15, min_periods=15).mean()

    vwr = price_change * vol_weight

    s = vwr.astype(float)
    s.name = "mom_vol_weighted_roc_15"
    return s

def mom_inertia_30(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Inertia over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    inertia = delta.rolling(30, min_periods=30).apply(lambda x: np.sum(x**2))

    s = inertia.astype(float)
    s.name = "mom_inertia_30"
    return s

def mom_cumulative_push_20(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Cumulative Push over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    cumulative_push = delta.rolling(20, min_periods=20).sum()

    s = cumulative_push.astype(float)
    s.name = "mom_cumulative_push_20"
    return s

def vol_range_volatility_20(df: pd.DataFrame) -> pd.Series:
    """
    Range-Based Volatility over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high_low_range = g["high"] - g["low"]
    rv = high_low_range.rolling(20, min_periods=20).std()

    s = rv.astype(float)
    s.name = "vol_range_volatility_20"
    return s

def vol_mean_reversion_ratio_30(df: pd.DataFrame) -> pd.Series:
    """
    Mean Reversion Ratio over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    mean_price = g["close"].rolling(30, min_periods=30).mean()
    mrr = (g["close"] - mean_price).abs() / mean_price.replace(0, np.nan)

    s = mrr.astype(float)
    s.name = "vol_mean_reversion_ratio_30"
    return s

def vol_vol_of_vol_50(df: pd.DataFrame) -> pd.Series:
    """
    Volatility of Volatility over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    vov = volatility.rolling(50, min_periods=50).std()

    s = vov.astype(float)
    s.name = "vol_vol_of_vol_50"
    return s


def vol_spike_density_25(df: pd.DataFrame) -> pd.Series:
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
    s.name = "vol_spike_density_25"
    return s

def vol_smooth_ratio_40(df: pd.DataFrame) -> pd.Series:
    """
    Volatility Smooth Ratio over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    smooth_vol = volatility.rolling(40, min_periods=40).mean()
    vsr = smooth_vol / volatility.replace(0, np.nan)

    s = vsr.astype(float)
    s.name = "vol_smooth_ratio_40"
    return s

def liq_depth_ratio_15(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Depth Ratio over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    avg_volume = g["volume"].rolling(15, min_periods=15).mean()
    price_range = g["high"] - g["low"]
    ldr = avg_volume / price_range.replace(0, np.nan)

    s = ldr.astype(float)
    s.name = "liq_depth_ratio_15"
    return s

def liq_volatility_corr_20(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity-Volatility Correlation over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    avg_volume = g["volume"].rolling(20, min_periods=20).mean()

    lvc = volatility.rolling(20, min_periods=20).corr(avg_volume)

    s = lvc.astype(float)
    s.name = "liq_volatility_corr_20"
    return s

def liq_pressure_index_25(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Pressure Index over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    buy_volume = g["volume"] * (g["close"] > g["open"]).astype(int)
    sell_volume = g["volume"] * (g["close"] < g["open"]).astype(int)

    lpi = (buy_volume - sell_volume).rolling(25, min_periods=25).sum() / g["volume"].rolling(25, min_periods=25).sum().replace(0, np.nan)

    s = lpi.astype(float)
    s.name = "liq_pressure_index_25"
    return s

def liq_absorption_ratio_30(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Absorption Ratio over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].diff()
    volume = g["volume"]

    lar = (price_change.abs() * volume).rolling(30, min_periods=30).sum() / volume.rolling(30, min_periods=30).sum().replace(0, np.nan)

    s = lar.astype(float)
    s.name = "liq_absorption_ratio_30"
    return s

def liq_turnover_rate_20(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Turnover Rate over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    turnover_rate = g["volume"] / g["volume"].rolling(20, min_periods=20).mean()

    s = turnover_rate.astype(float)
    s.name = "liq_turnover_rate_20"
    return s

def info_return_entropy_20(df: pd.DataFrame) -> pd.Series:
    """
    Information Return Entropy over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    returns = g["close"].pct_change().rolling(20, min_periods=20).apply(
        lambda x: -np.sum((x + 1) * np.log(x + 1 + 1e-10)) / np.log(len(x))
    )

    s = returns.astype(float)
    s.name = "info_return_entropy_20"
    return s

def info_volume_entropy_20(df: pd.DataFrame) -> pd.Series:
    """
    Information Volume Entropy over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    vol_changes = g["volume"].pct_change().rolling(20, min_periods=20).apply(
        lambda x: -np.sum((x + 1) * np.log(x + 1 + 1e-10)) / np.log(len(x))
    )

    s = vol_changes.astype(float)
    s.name = "info_volume_entropy_20"
    return s

def info_range_complexity_30(df: pd.DataFrame) -> pd.Series:
    """
    Information Range Complexity over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_range = g["high"] - g["low"]
    complexity = price_range.rolling(30, min_periods=30).apply(
        lambda x: -np.sum((x / x.sum()) * np.log((x / x.sum()) + 1e-10))
    )

    s = complexity.astype(float)
    s.name = "info_range_complexity_30"
    return s

def info_joint_entropy_20(df: pd.DataFrame) -> pd.Series:
    """
    Information Joint Entropy of Price and Volume over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_returns = g["close"].pct_change()
    vol_changes = g["volume"].pct_change()

    joint_entropy = pd.Series(index=g.index, dtype=float)

    for i in range(19, len(g)):
        pr = price_returns[i-19:i+1]
        vc = vol_changes[i-19:i+1]
        joint_dist = pd.crosstab(pr, vc, normalize=True)
        joint_entropy.iat[i] = -np.sum(joint_dist.values * np.log(joint_dist.values + 1e-10))

    joint_entropy.name = "info_joint_entropy_20"
    return joint_entropy

def info_directional_variability_25(df: pd.DataFrame) -> pd.Series:
    """
    Information Directional Variability over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    directional_changes = (delta > 0).astype(int) - (delta < 0).astype(int)

    variability = directional_changes.rolling(25, min_periods=25).apply(
        lambda x: -np.sum((x.value_counts(normalize=True) * np.log(x.value_counts(normalize=True) + 1e-10)))
    )

    s = variability.astype(float)
    s.name = "info_directional_variability_25"
    return s

def geo_body_wick_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Body to Wick Ratio
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    body = (g["close"] - g["open"]).abs()
    upper_wick = g["high"] - np.maximum(g["close"], g["open"])
    lower_wick = np.minimum(g["close"], g["open"]) - g["low"]
    total_wick = upper_wick + lower_wick

    gbwr = body / total_wick.replace(0, np.nan)

    s = gbwr.astype(float)
    s.name = "geo_body_wick_ratio"
    return s

def geo_direction_consistency_10(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Direction Consistency over 10 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    dir_consistency = delta.rolling(10, min_periods=10).apply(
        lambda x: np.prod(np.sign(x[x != 0])) if np.all(x != 0) else 0
    )

    s = dir_consistency.astype(float)
    s.name = "geo_direction_consistency_10"
    return s

def geo_price_position_20(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Price Position over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    min_price = g["low"].rolling(20, min_periods=20).min()
    max_price = g["high"].rolling(20, min_periods=20).max()

    price_pos = (g["close"] - min_price) / (max_price - min_price).replace(0, np.nan)

    s = price_pos.astype(float)
    s.name = "geo_price_position_20"
    return s

def geo_price_reversal_flag_15(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Price Reversal Flag over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    reversal_flag = delta.rolling(15, min_periods=15).apply(
        lambda x: 1 if (x[-1] * x[0] < 0) else 0
    )

    s = reversal_flag.astype(float)
    s.name = "geo_price_reversal_flag_15"
    return s

def geo_range_ratio_25(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Range Ratio over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high_low_range = g["high"] - g["low"]
    avg_range = high_low_range.rolling(25, min_periods=25).mean()

    range_ratio = high_low_range / avg_range.replace(0, np.nan)

    s = range_ratio.astype(float)
    s.name = "geo_range_ratio_25"
    return s

def geo_trend_strength_balance_30(df: pd.DataFrame) -> pd.Series:
    """
    Geometric Trend Strength Balance over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    up_moves = delta.clip(lower=0).rolling(30, min_periods=30).sum()
    down_moves = -delta.clip(upper=0).rolling(30, min_periods=30).sum()

    tsb = up_moves / down_moves.replace(0, np.nan)

    s = tsb.astype(float)
    s.name = "geo_trend_strength_balance_30"
    return s

def geo_struct_regime_score_50(df: pd.DataFrame) -> pd.Series:
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

    regime_score.name = "geo_struct_regime_score_50"
    return regime_score

def trend_swing_efficiency_30(df: pd.DataFrame) -> pd.Series:
    """
    Trend Swing Efficiency over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    swing_efficiency = delta.rolling(30, min_periods=30).apply(
        lambda x: np.sum(x[x > 0]) / -np.sum(x[x < 0]) if np.sum(x[x < 0]) != 0 else np.nan
    )

    s = swing_efficiency.astype(float)
    s.name = "trend_swing_efficiency_30"
    return s

def trend_channel_touch_freq_50(df: pd.DataFrame) -> pd.Series:
    """
    Trend Channel Touch Frequency over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high = g["high"]
    low = g["low"]
    close = g["close"]

    channel_touches = pd.Series(index=close.index, dtype=float)

    for i in range(49, len(close)):
        upper_channel = high[i-49:i+1].max()
        lower_channel = low[i-49:i+1].min()
        touches = ((close[i-49:i+1] >= upper_channel) | (close[i-49:i+1] <= lower_channel)).sum()
        channel_touches.iat[i] = touches / 50

    channel_touches.name = "trend_channel_touch_freq_50"
    return channel_touches

def trend_nonlin_slope_var_40(df: pd.DataFrame) -> pd.Series:
    """
    Trend Non-Linear Slope Variance over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    close = g["close"]
    slope_var = pd.Series(index=close.index, dtype=float)

    for i in range(39, len(close)):
        y = close[i-39:i+1]
        x = np.arange(40)
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        slope_var.iat[i] = np.var(residuals)

    slope_var.name = "trend_nonlin_slope_var_40"
    return slope_var

def mom_signed_energy_20(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Signed Energy over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    signed_energy = delta.rolling(20, min_periods=20).apply(
        lambda x: np.sum(x**2) * np.sign(x[-1])
    )

    s = signed_energy.astype(float)
    s.name = "mom_signed_energy_20"
    return s

def mom_persistence_index_25(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Persistence Index over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    persistence = delta.rolling(25, min_periods=25).apply(
        lambda x: np.sum(x > 0) / 25
    )

    s = persistence.astype(float)
    s.name = "mom_persistence_index_25"
    return s

def mom_entropy_weighted_roc_15(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Entropy-Weighted Rate of Change over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].pct_change(15)
    entropy = -price_change.rolling(15, min_periods=15).apply(
        lambda x: np.sum((x + 1) * np.log(x + 1 + 1e-10)) / np.log(len(x))
    )

    ewr = price_change * entropy

    s = ewr.astype(float)
    s.name = "mom_entropy_weighted_roc_15"
    return s

def vol_dynamic_range_index_30(df: pd.DataFrame) -> pd.Series:
    """
    Volatility Dynamic Range Index over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    high_low_range = g["high"] - g["low"]
    dri = high_low_range.rolling(30, min_periods=30).apply(
        lambda x: (x.max() - x.min()) / x.mean() if x.mean() != 0 else np.nan
    )

    s = dri.astype(float)
    s.name = "vol_dynamic_range_index_30"
    return s

def vol_cluster_density_40(df: pd.DataFrame) -> pd.Series:
    """
    Volatility Cluster Density over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    mean_vol = volatility.rolling(40, min_periods=40).mean()

    cluster_density = (volatility > mean_vol).rolling(40, min_periods=40).mean()

    s = cluster_density.astype(float)
    s.name = "vol_cluster_density_40"
    return s

def liq_flow_balance_20(df: pd.DataFrame) -> pd.Series:
    """
    Liquidity Flow Balance over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    buy_volume = g["volume"] * (g["close"] > g["open"]).astype(int)
    sell_volume = g["volume"] * (g["close"] < g["open"]).astype(int)

    lfb = (buy_volume - sell_volume).rolling(20, min_periods=20).sum()

    s = lfb.astype(float)
    s.name = "liq_flow_balance_20"
    return s

def liq_spike_persistence_15(df: pd.DataFrame) -> pd.Series:
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
    s.name = "liq_spike_persistence_15"
    return s

def info_return_direction_entropy_40(df: pd.DataFrame) -> pd.Series:
    """
    Information Return Direction Entropy over 40 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    dir_changes = (delta > 0).astype(int) - (delta < 0).astype(int)

    direction_entropy = dir_changes.rolling(40, min_periods=40).apply(
        lambda x: -np.sum((x.value_counts(normalize=True) * np.log(x.value_counts(normalize=True) + 1e-10)))
    )

    s = direction_entropy.astype(float)
    s.name = "info_return_direction_entropy_40"
    return s

def info_cross_entropy_price_volume_30(df: pd.DataFrame) -> pd.Series:
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

    cross_entropy.name = "info_cross_entropy_price_volume_30"
    return cross_entropy

def hybrid_vol_mom_regime_score_50(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Volatility-Momentum Regime Score over 50 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    momentum = g["close"].diff(10)

    regime_score = (momentum / volatility.replace(0, np.nan)).rolling(50, min_periods=50).mean()

    s = regime_score.astype(float)
    s.name = "hybrid_vol_mom_regime_score_50"
    return s

def hybrid_volume_trend_conflict_20(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Volume-Trend Conflict Indicator over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    trend = delta.rolling(20, min_periods=20).mean()
    avg_volume = g["volume"].rolling(20, min_periods=20).mean()

    conflict = (trend * (avg_volume - avg_volume.mean())).rolling(20, min_periods=20).mean()

    s = conflict.astype(float)
    s.name = "hybrid_volume_trend_conflict_20"
    return s

def hybrid_price_absorption_strength_25(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Price Absorption Strength over 25 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].diff()
    volume = g["volume"]

    absorption_strength = (price_change.abs() * volume).rolling(25, min_periods=25).sum() / volume.rolling(25, min_periods=25).sum().replace(0, np.nan)

    s = absorption_strength.astype(float)
    s.name = "hybrid_price_absorption_strength_25"
    return s

def hybrid_entropy_trend_momentum_30(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Entropy-Trend-Momentum Indicator over 30 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    delta = g["close"].diff()
    trend = delta.rolling(30, min_periods=30).mean()
    momentum = delta.rolling(30, min_periods=30).std()

    entropy = -trend.rolling(30, min_periods=30).apply(
        lambda x: np.sum((x / x.sum()) * np.log((x / x.sum()) + 1e-10))
    )

    htm = (trend * momentum * entropy).rolling(30, min_periods=30).mean()

    s = htm.astype(float)
    s.name = "hybrid_entropy_trend_momentum_30"
    return s

def hybrid_liq_volatility_interplay_20(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Liquidity-Volatility Interplay over 20 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    volatility = g["close"].rolling(20, min_periods=20).std()
    avg_volume = g["volume"].rolling(20, min_periods=20).mean()

    lvi = (volatility * avg_volume).rolling(20, min_periods=20).mean()

    s = lvi.astype(float)
    s.name = "hybrid_liq_volatility_interplay_20"
    return s

def hybrid_price_volume_momentum_15(df: pd.DataFrame) -> pd.Series:
    """
    Hybrid Price-Volume-Momentum Indicator over 15 periods.
    """
    g = df.copy()
    g.columns = [c.lower() for c in g.columns]

    price_change = g["close"].pct_change(15)
    vol_change = g["volume"].pct_change(15)
    momentum = g["close"].diff(15)

    pvm = (price_change * vol_change * momentum).rolling(15, min_periods=15).mean()

    s = pvm.astype(float)
    s.name = "hybrid_price_volume_momentum_15"
    return s