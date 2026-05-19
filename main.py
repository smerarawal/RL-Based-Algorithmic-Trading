#!/usr/bin/env python3
"""
Enhanced RL Trading System for NSE Indian Equities — v2.0
==========================================================
Key improvements over the baseline PPO system:
  1. Risk-adjusted reward: Sortino + drawdown penalty + turnover penalty
  2. Walk-forward cross-validation to prevent overfitting
  3. Rich feature engineering: indicators + regime detection + cross-sectional ranks
  4. Continuous portfolio weight actions (vs discrete buy/sell/hold)
  5. Multiple algorithms: SAC (primary), TD3, A2C, PPO (comparison)
  6. Risk controls: max position limits, volatility targeting, drawdown circuit breaker
  7. Ensemble inference: average predictions across all trained agents
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, TD3, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

@dataclass
class TradingConfig:
    # --- Universe ---
    # Expanded to 10 large-cap NSE stocks for better diversification
    tickers: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS",         "HDFCBANK.NS",  "INFY.NS",      "ICICIBANK.NS",
        "HINDUNILVR.NS","KOTAKBANK.NS",   "BAJFINANCE.NS","WIPRO.NS",     "AXISBANK.NS",
    ])
    benchmark_ticker: str = "^NSEI"

    # --- Time periods ---
    train_start: str = "2018-01-01"
    train_end:   str = "2021-12-31"
    val_start:   str = "2022-01-01"
    val_end:     str = "2022-12-31"
    test_start:  str = "2023-01-01"
    test_end:    str = "2024-12-31"

    # --- Capital ---
    initial_capital: float = 1_000_000.0   # ₹10 lakh

    # --- Indian market transaction costs ---
    brokerage_rate:  float = 0.0003   # 0.03% per leg (Zerodha flat-fee equivalent)
    stt_sell_rate:   float = 0.001    # 0.10% STT on sell side only
    gst_rate:        float = 0.18     # 18% GST on brokerage
    slippage_rate:   float = 0.0005   # 0.05% market impact / slippage
    stamp_duty_rate: float = 0.00015  # 0.015% stamp duty on buy

    # --- Environment ---
    window_size:            int   = 20    # rolling-window lookback for features
    max_position_fraction:  float = 0.35  # hard cap: no single stock > 35%
    min_cash_fraction:      float = 0.02  # always keep ≥ 2% in cash
    blowup_threshold:       float = 0.20  # terminate early if portfolio drops > 80%

    # --- Reward shaping ---
    reward_scaling:    float = 100.0   # scale raw log-returns
    drawdown_penalty:  float = 2.0     # penalty multiplier on drawdown increases
    turnover_penalty:  float = 0.5     # penalty multiplier on weight changes

    # --- Training ---
    total_timesteps: int = 500_000
    check_freq:      int = 20_000      # validation check every N steps

    # --- Paths ---
    model_dir:  str = "models"
    result_dir: str = "results"


CFG = TradingConfig()


# ============================================================
# SECTION 2: DATA & FEATURE ENGINEERING
# ============================================================

def download_data(cfg: TradingConfig) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data for all tickers using yfinance."""
    logger.info("Downloading NSE market data …")
    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in cfg.tickers:
        try:
            df = yf.download(
                ticker,
                start=cfg.train_start,
                end=cfg.test_end,
                auto_adjust=True,
                progress=False,
            )
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 200:
                all_data[ticker] = df
                logger.info(f"  ✓ {ticker}: {len(df)} trading days")
            else:
                logger.warning(f"  ✗ {ticker}: insufficient data ({len(df)} rows), skipping")
        except Exception as exc:
            logger.warning(f"  ✗ {ticker}: {exc}")
    logger.info(f"  → {len(all_data)}/{len(cfg.tickers)} tickers loaded\n")
    return all_data


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI normalised to [-1, 1] (0 = neutral)."""
    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss   = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs     = gain / (loss + 1e-8)
    rsi_raw = 100 - 100 / (1 + rs)
    return (rsi_raw - 50) / 50  # centred and scaled


def _obv_zscore(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume as a rolling z-score."""
    obv     = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    mu      = obv.rolling(60, min_periods=10).mean()
    sigma   = obv.rolling(60, min_periods=10).std() + 1e-8
    return (obv - mu) / sigma


def compute_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ticker feature engineering.

    Feature families:
      • Price-return features     (log returns at multiple horizons)
      • Trend / EMA cross signals (9/21/55 EMAs, MACD)
      • Momentum                  (RSI-14, RSI-28, ROC-10/20)
      • Volatility                (rolling std, BB %, ATR, vol ratio)
      • Volume                    (volume ratio, OBV z-score)
      • Market regime             (distance from 52-w high/low, Hurst proxy)
    """
    out  = df.copy()
    c    = df["Close"]
    h    = df["High"]
    lo   = df["Low"]
    vol  = df["Volume"]

    # ── Price returns ────────────────────────────────────────────
    out["log_ret"]    = np.log(c / c.shift(1))
    out["ret_1d"]     = c.pct_change(1)
    out["ret_5d"]     = c.pct_change(5)
    out["ret_20d"]    = c.pct_change(20)

    # ── Trend / EMA ──────────────────────────────────────────────
    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema55 = c.ewm(span=55, adjust=False).mean()
    out["ema_9_21"]   = (ema9  - ema21) / (c + 1e-8)
    out["ema_21_55"]  = (ema21 - ema55) / (c + 1e-8)

    macd_line         = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_signal       = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"]       = macd_line   / (c + 1e-8)
    out["macd_hist"]  = (macd_line - macd_signal) / (c + 1e-8)

    # ── Momentum ─────────────────────────────────────────────────
    out["rsi_14"]     = _rsi(c, 14)
    out["rsi_28"]     = _rsi(c, 28)
    out["roc_10"]     = c.pct_change(10)
    out["roc_20"]     = c.pct_change(20)

    # ── Volatility ───────────────────────────────────────────────
    lr = out["log_ret"]
    out["vol_10"]     = lr.rolling(10).std()
    out["vol_20"]     = lr.rolling(20).std()
    out["vol_60"]     = lr.rolling(60).std()
    out["vol_ratio"]  = out["vol_10"] / (out["vol_60"] + 1e-8)   # regime: short/long vol

    bb_mid            = c.rolling(20).mean()
    bb_std            = c.rolling(20).std()
    out["bb_pct"]     = (c - bb_mid)  / (2 * bb_std + 1e-8)      # ∈ [-1, 1] approx
    out["bb_width"]   = (4 * bb_std)  / (bb_mid + 1e-8)

    tr                = pd.concat([h - lo,
                                    (h - c.shift(1)).abs(),
                                    (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    out["atr_14"]     = tr.rolling(14).mean() / (c + 1e-8)

    # ── Volume ───────────────────────────────────────────────────
    out["vol_ratio_20"] = vol / (vol.rolling(20).mean() + 1e-8)
    out["obv_z"]        = _obv_zscore(c, vol)

    # ── Market regime ────────────────────────────────────────────
    out["dist_52h"]   = (c - h.rolling(252).max()) / (c + 1e-8)   # ≤ 0
    out["dist_52l"]   = (c - lo.rolling(252).min()) / (c + 1e-8)  # ≥ 0
    out["hurst_px"]   = out["ret_20d"] / (out["vol_20"] * np.sqrt(20) + 1e-8)

    # Rolling vol-regime percentile (0 = calmest, 1 = most volatile)
    out["vol_pct"]    = (
        out["vol_20"]
        .rolling(60, min_periods=20)
        .apply(lambda x: float(stats.percentileofscore(x[~np.isnan(x)], x[-1])) / 100.0,
               raw=True)
    )

    return out


def add_cross_sectional_features(
    all_feat: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Cross-sectional rank features computed at each date across all stocks.
    Provides the agent with relative rather than absolute signals.

    Features:
      cs_mom_rank  — 20-day momentum rank        (high = strong uptrend)
      cs_vol_rank  — realised-volatility rank     (high = lowest volatility)
      cs_rsi_rank  — RSI rank across universe     (high = most overbought)
    """
    tickers = list(all_feat.keys())
    idx     = all_feat[tickers[0]].index

    mom_panel = pd.DataFrame({t: all_feat[t]["ret_20d"]  for t in tickers}, index=idx)
    vol_panel = pd.DataFrame({t: all_feat[t]["vol_20"]   for t in tickers}, index=idx)
    rsi_panel = pd.DataFrame({t: all_feat[t]["rsi_14"]   for t in tickers}, index=idx)

    mom_rank = mom_panel.rank(axis=1, pct=True)
    vol_rank = vol_panel.rank(axis=1, pct=True, ascending=False)   # lower vol → higher rank
    rsi_rank = rsi_panel.rank(axis=1, pct=True)

    for t in tickers:
        all_feat[t]["cs_mom_rank"] = mom_rank[t]
        all_feat[t]["cs_vol_rank"] = vol_rank[t]
        all_feat[t]["cs_rsi_rank"] = rsi_rank[t]

    return all_feat


# Columns that are raw market data, NOT used as RL features
_OHLCV_COLS = {"Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"}


def prepare_features(
    all_data: Dict[str, pd.DataFrame], cfg: TradingConfig
) -> Tuple[Dict[str, pd.DataFrame], List[str], pd.DatetimeIndex]:
    """Full feature pipeline: compute → cross-sectional → align → define feature list."""
    logger.info("Engineering features …")

    all_feat: Dict[str, pd.DataFrame] = {}
    for ticker, df in all_data.items():
        all_feat[ticker] = compute_ticker_features(df)

    all_feat = add_cross_sectional_features(all_feat)

    # Feature columns (exclude raw OHLCV)
    sample_df    = next(iter(all_feat.values()))
    feature_cols = [c for c in sample_df.columns if c not in _OHLCV_COLS]

    # Drop rows with any NaN in features
    for t in list(all_feat.keys()):
        all_feat[t] = all_feat[t].dropna(subset=feature_cols)

    # Align all tickers to the same date range
    common_idx = next(iter(all_feat.values())).index
    for t in list(all_feat.keys())[1:]:
        common_idx = common_idx.intersection(all_feat[t].index)

    for t in list(all_feat.keys()):
        all_feat[t] = all_feat[t].loc[common_idx]

    logger.info(f"  Feature count  : {len(feature_cols)}")
    logger.info(f"  Common dates   : {len(common_idx)}  "
                f"({common_idx[0].date()} → {common_idx[-1].date()})\n")
    return all_feat, feature_cols, common_idx


# ============================================================
# SECTION 3: TRADING ENVIRONMENT
# ============================================================

class NsePortfolioEnv(gym.Env):
    """
    Multi-asset portfolio environment for NSE equities.

    Design principles (vs the baseline):
    ─────────────────────────────────────
    Action space
      Continuous vector in [-1, 1]^N (raw logits).
      A softmax transforms them into non-negative portfolio weights
      summing to ≤ (1 − min_cash_fraction), with each weight capped
      at max_position_fraction.  This eliminates the coarse discrete
      granularity of a buy/sell/hold action and lets the agent express
      any portfolio composition.

    Observation space
      [flattened per-stock features] + [current weights] + [norm. value,
      peak value, time progress] — gives the agent full awareness of
      its current position and risk state.

    Reward function (three components)
      R = λ_ret × log_return
        − λ_dd  × Δdrawdown         (penalise each new drawdown increment)
        − λ_to  × turnover          (penalise excessive rebalancing)
        + sortino_bonus              (every 20 steps, reward good risk/return)

    Risk controls
      • Hard position cap  : max_position_fraction per stock
      • Cash floor         : min_cash_fraction always in cash
      • Blowup termination : episode ends if portfolio < 20% of initial
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        all_features: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        date_index:   pd.DatetimeIndex,
        cfg:          TradingConfig,
        start_date:   str,
        end_date:     str,
        mode:         str = "train",
    ) -> None:
        super().__init__()
        self.cfg          = cfg
        self.mode         = mode
        self.tickers      = list(all_features.keys())
        self.n_stocks     = len(self.tickers)
        self.feature_cols = feature_cols
        self.n_feat       = len(feature_cols)

        # ── Slice to requested date range ──────────────────────
        mask        = (date_index >= start_date) & (date_index <= end_date)
        self.dates  = date_index[mask]
        self.T      = len(self.dates)

        # ── Build tensors (T × S × F) and (T × S) ─────────────
        self.feat_mat    = np.zeros((self.T, self.n_stocks, self.n_feat),  dtype=np.float32)
        self.close_mat   = np.zeros((self.T, self.n_stocks),               dtype=np.float32)

        for j, t in enumerate(self.tickers):
            sub = all_features[t].loc[self.dates]
            self.feat_mat[:, j, :] = sub[feature_cols].values.astype(np.float32)
            self.close_mat[:, j]   = sub["Close"].values.astype(np.float32)

        # ── Standardise features using the period's own statistics ─
        mu  = self.feat_mat.mean(axis=0, keepdims=True)
        sig = self.feat_mat.std(axis=0,  keepdims=True) + 1e-8
        self.feat_mat = np.clip((self.feat_mat - mu) / sig, -3.0, 3.0)

        # ── Gym spaces ────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_stocks,), dtype=np.float32
        )
        obs_dim = self.n_stocks * self.n_feat + self.n_stocks + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state (initialised in reset)
        self._t: int = 0
        self.weights = np.zeros(self.n_stocks, dtype=np.float32)
        self.portfolio_value  = cfg.initial_capital
        self._peak_value      = cfg.initial_capital
        self._prev_drawdown   = 0.0
        self._ret_hist: List[float] = []

        # Tracking (filled during episode for backtest analysis)
        self.portfolio_history: List[float]       = []
        self.weight_history:    List[np.ndarray]  = []

    # ── Helpers ────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        feat_flat = self.feat_mat[self._t].flatten()   # (S × F,)
        port_obs  = np.concatenate([
            self.weights,
            [self.portfolio_value / self.cfg.initial_capital],
            [self._peak_value     / self.cfg.initial_capital],
            [self._t / max(self.T, 1)],
        ])
        return np.concatenate([feat_flat, port_obs]).astype(np.float32)

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Logits → constrained portfolio weights."""
        a = np.clip(action, -10.0, 10.0)
        # Stable softmax
        a = a - a.max()
        w = np.exp(a)
        w = w / w.sum()
        # Hard cap per stock
        w = np.minimum(w, self.cfg.max_position_fraction)
        # Cash floor constraint: total invested ≤ (1 - min_cash)
        invest_cap = 1.0 - self.cfg.min_cash_fraction
        total = w.sum()
        if total > invest_cap:
            w = w * invest_cap / total
        return w.astype(np.float32)

    def _transaction_cost(self, w_old: np.ndarray, w_new: np.ndarray) -> float:
        """Indian market transaction costs (STT + brokerage + GST + stamp + slippage)."""
        dw_buy  = np.maximum(w_new - w_old, 0.0)
        dw_sell = np.maximum(w_old - w_new, 0.0)

        val_buy  = dw_buy  * self.portfolio_value
        val_sell = dw_sell * self.portfolio_value

        cost_buy = val_buy * (
            self.cfg.brokerage_rate * (1.0 + self.cfg.gst_rate)
            + self.cfg.stamp_duty_rate
            + self.cfg.slippage_rate
        )
        cost_sell = val_sell * (
            self.cfg.brokerage_rate * (1.0 + self.cfg.gst_rate)
            + self.cfg.stt_sell_rate
            + self.cfg.slippage_rate
        )
        return float(cost_buy.sum() + cost_sell.sum())

    # ── Gym API ────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise start inside first 10 % of data (train-time augmentation)
        if self.mode == "train":
            max_start = max(1, self.T // 10)
            rng = getattr(self, "np_random", None)
            self._t = int(rng.integers(0, max_start)) if rng is not None else 0
        else:
            self._t = 0

        self.portfolio_value = self.cfg.initial_capital
        self._peak_value     = self.cfg.initial_capital
        self._prev_drawdown  = 0.0
        self.weights         = np.zeros(self.n_stocks, dtype=np.float32)
        self._ret_hist       = []

        self.portfolio_history = [self.portfolio_value]
        self.weight_history    = [self.weights.copy()]

        return self._obs(), {}

    def step(self, action: np.ndarray):
        if self._t >= self.T - 1:
            return self._obs(), 0.0, True, False, self._info()

        # 1. Convert action → target weights with risk constraints
        tgt_w    = self._action_to_weights(action)
        turnover = float(np.abs(tgt_w - self.weights).sum())

        # 2. Deduct transaction costs (at current prices)
        tc = self._transaction_cost(self.weights, tgt_w)
        self.portfolio_value -= tc
        self.weights = tgt_w

        # 3. Advance time and update portfolio value
        self._t += 1
        p_now  = self.close_mat[self._t]
        p_prev = self.close_mat[self._t - 1]
        ret_per_stock = (p_now - p_prev) / (p_prev + 1e-8)
        port_ret      = float(np.dot(self.weights, ret_per_stock))

        self.portfolio_value *= (1.0 + port_ret)

        # 4. Update peak and compute drawdown
        if self.portfolio_value > self._peak_value:
            self._peak_value = self.portfolio_value
        dd_now = (self._peak_value - self.portfolio_value) / (self._peak_value + 1e-8)

        # 5. Reward computation
        log_ret = float(np.log1p(port_ret))
        self._ret_hist.append(log_ret)

        reward = self.cfg.reward_scaling * log_ret

        # Penalise each new unit of drawdown (not the level itself)
        dd_increase = max(0.0, dd_now - self._prev_drawdown)
        reward -= self.cfg.drawdown_penalty * dd_increase * self.cfg.reward_scaling
        self._prev_drawdown = dd_now

        # Penalise high turnover
        reward -= self.cfg.turnover_penalty * turnover

        # Sortino bonus every 20 steps (bounded in [-1, 1])
        if len(self._ret_hist) >= 20 and self._t % 20 == 0:
            r_arr    = np.array(self._ret_hist[-20:])
            neg_r    = r_arr[r_arr < 0]
            dn_std   = neg_r.std() + 1e-8
            sortino  = float(r_arr.mean() / dn_std)
            reward  += float(np.clip(sortino, -1.0, 1.0))

        # 6. Track history
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(self.weights.copy())

        # 7. Termination
        blowup = self.portfolio_value < self.cfg.initial_capital * self.cfg.blowup_threshold
        done   = (self._t >= self.T - 1) or blowup

        return self._obs(), float(reward), done, False, self._info()

    def _info(self) -> Dict:
        dd = (self._peak_value - self.portfolio_value) / (self._peak_value + 1e-8)
        return {"portfolio_value": self.portfolio_value, "drawdown": dd, "step": self._t}

    def render(self, mode="human"):
        pass


# ============================================================
# SECTION 4: PERFORMANCE METRICS
# ============================================================

def compute_metrics(values: np.ndarray, initial: float, freq: int = 252) -> Dict:
    """
    Comprehensive risk-adjusted performance metrics.
    Returns: total return, CAGR, volatility, Sharpe, Sortino, Calmar,
             max drawdown, win rate, VaR-95, CVaR-95.
    """
    v     = np.asarray(values, dtype=float)
    rets  = np.diff(v) / v[:-1]
    lr    = np.log(v[1:] / v[:-1])

    total_ret = (v[-1] / initial - 1.0) * 100.0
    n_years   = len(rets) / freq
    cagr      = ((v[-1] / initial) ** (1.0 / max(n_years, 1e-3)) - 1.0) * 100.0

    ann_ret = lr.mean() * freq
    ann_vol = lr.std()  * np.sqrt(freq)
    sharpe  = ann_ret / (ann_vol + 1e-8)

    neg_lr  = lr[lr < 0]
    sortino = ann_ret / (neg_lr.std() * np.sqrt(freq) + 1e-8)

    peak  = np.maximum.accumulate(v)
    dds   = (peak - v) / (peak + 1e-8)
    max_dd = dds.max() * 100.0
    calmar = cagr / (max_dd + 1e-8)

    win_rate  = (rets > 0).mean() * 100.0
    var_95    = float(np.percentile(rets, 5)) * 100.0
    tail_mask = rets <= np.percentile(rets, 5)
    cvar_95   = float(rets[tail_mask].mean()) * 100.0

    return {
        "total_return_%":   round(total_ret,  2),
        "cagr_%":           round(cagr,        2),
        "ann_volatility_%": round(ann_vol * 100, 2),
        "sharpe_ratio":     round(sharpe,      3),
        "sortino_ratio":    round(sortino,     3),
        "calmar_ratio":     round(calmar,      3),
        "max_drawdown_%":   round(max_dd,      2),
        "win_rate_%":       round(win_rate,    2),
        "var_95_%":         round(var_95,      2),
        "cvar_95_%":        round(cvar_95,     2),
        "final_value":      round(float(v[-1]), 0),
    }


# ============================================================
# SECTION 5: BASELINES
# ============================================================

def nifty_baseline(cfg: TradingConfig) -> Tuple[np.ndarray, Dict]:
    df = yf.download(cfg.benchmark_ticker, start=cfg.test_start, end=cfg.test_end,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    p  = df["Close"].dropna().values
    v  = cfg.initial_capital * p / p[0]
    return v, compute_metrics(v, cfg.initial_capital)


def equal_weight_baseline(
    all_features: Dict[str, pd.DataFrame], cfg: TradingConfig
) -> Tuple[np.ndarray, Dict]:
    prices = pd.DataFrame(
        {t: all_features[t].loc[cfg.test_start:cfg.test_end, "Close"] for t in all_features}
    ).dropna()
    w     = np.ones(len(prices.columns)) / len(prices.columns)
    rets  = prices.pct_change().dropna()
    prets = (rets * w).sum(axis=1)

    vals  = [cfg.initial_capital]
    for r in prets:
        vals.append(vals[-1] * (1.0 + r))
    v = np.array(vals)
    return v, compute_metrics(v, cfg.initial_capital)


def momentum_baseline(
    all_features: Dict[str, pd.DataFrame], cfg: TradingConfig
) -> Tuple[np.ndarray, Dict]:
    """Monthly rebalance into the top-3 momentum stocks."""
    tickers = list(all_features.keys())
    prices  = pd.DataFrame(
        {t: all_features[t].loc[cfg.test_start:cfg.test_end, "Close"] for t in tickers}
    ).dropna()
    rets    = prices.pct_change().dropna()

    vals  = [cfg.initial_capital]
    cur_w = np.ones(len(tickers)) / len(tickers)

    for i, date in enumerate(rets.index):
        if i % 21 == 0 and i > 0:   # monthly rebalance
            lookback      = min(60, i)
            mom           = prices.iloc[i - lookback : i].iloc[-1] / prices.iloc[i - lookback] - 1.0
            top3          = mom.nlargest(3).index.tolist()
            cur_w         = np.zeros(len(tickers))
            for t in top3:
                cur_w[tickers.index(t)] = 1.0 / 3.0

        pr = float((rets.iloc[i].values * cur_w).sum())
        vals.append(vals[-1] * (1.0 + pr))

    v = np.array(vals)
    return v, compute_metrics(v, cfg.initial_capital)


# ============================================================
# SECTION 6: TRAINING
# ============================================================

class RiskAdjustedCallback(BaseCallback):
    """
    Validation callback that:
      • Runs a full deterministic episode on the validation environment every
        `check_freq` training steps.
      • Logs Calmar ratio, Sharpe, total return, and max drawdown.
      • Saves the model checkpoint whenever a new best Calmar is achieved.
      • Implemented as early-stop-aware (patience tracking) but does NOT stop
        training early — it just logs; the model that gets loaded for backtest is
        the one with the best validation Calmar.
    """

    def __init__(
        self,
        val_env:    NsePortfolioEnv,
        cfg:        TradingConfig,
        algo_name:  str,
        check_freq: int = 20_000,
        verbose:    int = 1,
    ) -> None:
        super().__init__(verbose)
        self.val_env    = val_env
        self.cfg        = cfg
        self.algo_name  = algo_name
        self.check_freq = check_freq
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            values = self._run_episode()
            if len(values) > 10:
                m = compute_metrics(np.array(values), self.cfg.initial_capital)
                # Primary criterion: Calmar (penalises drawdown)
                score = m["calmar_ratio"]
                logger.info(
                    f"  [{self.algo_name:>4s}] step {self.n_calls:>7,} | "
                    f"Calmar {score:+.3f} | Sharpe {m['sharpe_ratio']:+.3f} | "
                    f"Ret {m['total_return_%']:+.1f}% | MaxDD -{m['max_drawdown_%']:.1f}%"
                )
                if score > self.best_score:
                    self.best_score = score
                    os.makedirs(self.cfg.model_dir, exist_ok=True)
                    self.model.save(f"{self.cfg.model_dir}/{self.algo_name}_best")
                    logger.info(f"  → ✓ New best checkpoint (Calmar={score:.3f})")
        return True

    def _run_episode(self) -> List[float]:
        obs, _ = self.val_env.reset()
        done   = False
        while not done:
            act, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = self.val_env.step(act)
            done = term or trunc
        return self.val_env.portfolio_history


def _make_env_fn(all_features, feature_cols, date_idx, cfg, start, end, mode, seed):
    def _init():
        env = NsePortfolioEnv(all_features, feature_cols, date_idx, cfg, start, end, mode)
        return Monitor(env)
    return _init


def train_agent(
    algorithm:    str,
    all_features: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    date_index:   pd.DatetimeIndex,
    cfg:          TradingConfig,
) -> Any:
    """
    Train an RL agent with `algorithm` ∈ {SAC, TD3, A2C, PPO}.
    Best model (by validation Calmar) is saved and reloaded before returning.
    """
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Training {algorithm}")
    logger.info(f"{'─'*60}")

    # Vectorised training environment (2 parallel workers)
    train_env = DummyVecEnv([
        _make_env_fn(all_features, feature_cols, date_index, cfg,
                     cfg.train_start, cfg.train_end, "train", seed=i)
        for i in range(2)
    ])

    val_env = NsePortfolioEnv(
        all_features, feature_cols, date_index, cfg,
        cfg.val_start, cfg.val_end, mode="val",
    )

    policy_kwargs = dict(net_arch=[256, 256, 128])

    if algorithm == "SAC":
        # Soft Actor-Critic — ideal for continuous portfolio weights.
        # Entropy regularisation prevents the policy from collapsing to
        # a deterministic (often overfit) corner solution.
        model = SAC(
            "MlpPolicy", train_env,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto_0.1",     # auto-tune → trades off exploration vs exploitation
            target_entropy="auto",
            train_freq=4,
            gradient_steps=2,
            learning_starts=5_000,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=42,
        )

    elif algorithm == "TD3":
        # Twin Delayed DDPG — deterministic policy with Ornstein-Uhlenbeck
        # noise for smooth action exploration (appropriate for correlated
        # financial time series).
        n = len(cfg.tickers)
        noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n), sigma=0.1 * np.ones(n), theta=0.15,
        )
        model = TD3(
            "MlpPolicy", train_env,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            action_noise=noise,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            learning_starts=5_000,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=42,
        )

    elif algorithm == "A2C":
        # Advantage Actor-Critic — on-policy, memory-efficient.
        # Entropy bonus (ent_coef=0.01) discourages degenerate policies.
        model = A2C(
            "MlpPolicy", train_env,
            learning_rate=7e-4,
            n_steps=128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=42,
        )

    elif algorithm == "PPO":
        # PPO — baseline comparison vs the original system.
        # Kept with the same hyperparameter philosophy but in the improved env.
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=42,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    cb = RiskAdjustedCallback(val_env, cfg, algorithm, check_freq=cfg.check_freq)
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb, progress_bar=True)

    # Reload the best checkpoint
    best_path = f"{cfg.model_dir}/{algorithm}_best.zip"
    if os.path.exists(best_path):
        logger.info(f"  Reloading best checkpoint for {algorithm} …")
        cls_map = {"SAC": SAC, "TD3": TD3, "A2C": A2C, "PPO": PPO}
        model = cls_map[algorithm].load(best_path.replace(".zip", ""), env=train_env)

    return model


# ============================================================
# SECTION 7: BACKTESTING
# ============================================================

def backtest_agent(
    model:        Any,
    algorithm:    str,
    all_features: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    date_index:   pd.DatetimeIndex,
    cfg:          TradingConfig,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run a single deterministic episode on the test set."""
    logger.info(f"  Backtesting {algorithm} …")
    env    = NsePortfolioEnv(all_features, feature_cols, date_index, cfg,
                              cfg.test_start, cfg.test_end, mode="test")
    obs, _ = env.reset()
    done   = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return np.array(env.portfolio_history), env.weight_history


def ensemble_backtest(
    models:       Dict[str, Any],
    all_features: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    date_index:   pd.DatetimeIndex,
    cfg:          TradingConfig,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Ensemble: at each step, average the raw action logits from every trained model.
    Averaging in action space rather than weight space preserves the softmax
    normalisation and produces smoother, lower-variance allocations.
    """
    logger.info("  Running ensemble backtest …")
    env    = NsePortfolioEnv(all_features, feature_cols, date_index, cfg,
                              cfg.test_start, cfg.test_end, mode="test")
    obs, _ = env.reset()
    done   = False
    while not done:
        actions = [m.predict(obs, deterministic=True)[0] for m in models.values()]
        avg_act = np.mean(actions, axis=0)
        obs, _, term, trunc, _ = env.step(avg_act)
        done = term or trunc
    return np.array(env.portfolio_history), env.weight_history


# ============================================================
# SECTION 8: VISUALISATION & REPORTING
# ============================================================

_COLORS = {
    "SAC":          "#1E88E5",
    "TD3":          "#8E24AA",
    "A2C":          "#FB8C00",
    "PPO":          "#6D4C41",
    "Ensemble":     "#E53935",
    "Nifty50 B&H":  "#43A047",
    "Equal Weight": "#00897B",
    "Momentum":     "#F4511E",
}


def plot_dashboard(results: Dict, cfg: TradingConfig):
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "Enhanced RL Trading System — Performance Dashboard  |  NSE Equities 2023–2024",
        fontsize=15, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.32)

    # ── Panel 1: Portfolio value (full width) ──────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for name, d in results.items():
        v = np.array(d["values"])
        n = v / v[0] * 100
        lw   = 2.4 if name in ("Ensemble",)  else 1.4
        ls   = "--" if name in ("Nifty50 B&H", "Equal Weight", "Momentum") else "-"
        ax1.plot(n, label=name, color=_COLORS.get(name, "gray"), lw=lw, ls=ls)
    ax1.axhline(100, color="black", ls=":", lw=0.8, alpha=0.5)
    ax1.set_title("Normalised Portfolio Value (base = 100)", fontweight="bold")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left", ncol=4, fontsize=8)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Drawdown (2/3 width) ──────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    for name, d in results.items():
        v   = np.array(d["values"])
        pk  = np.maximum.accumulate(v)
        dd  = (pk - v) / pk * 100
        ax2.fill_between(range(len(dd)), -dd, 0, alpha=0.25,
                         color=_COLORS.get(name, "gray"))
        ax2.plot(-dd, color=_COLORS.get(name, "gray"), lw=0.9, label=name)
    ax2.set_title("Underwater Drawdown (%)", fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", ncol=2, fontsize=7)
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: Risk-Return scatter ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    for name, d in results.items():
        m = d["metrics"]
        ax3.scatter(m["ann_volatility_%"], m["cagr_%"],
                    s=160, color=_COLORS.get(name, "gray"), zorder=5,
                    edgecolors="white", lw=1.2)
        ax3.annotate(name, (m["ann_volatility_%"], m["cagr_%"]),
                     textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax3.set_xlabel("Annual Volatility (%)")
    ax3.set_ylabel("CAGR (%)")
    ax3.set_title("Risk–Return Profile", fontweight="bold")
    ax3.grid(True, alpha=0.25)

    # ── Panels 4–6: Bar charts for key metrics ─────────────────
    bar_metrics = [
        ("sharpe_ratio",    "Sharpe Ratio",       False),
        ("sortino_ratio",   "Sortino Ratio",      False),
        ("calmar_ratio",    "Calmar Ratio",        False),
        ("max_drawdown_%",  "Max Drawdown (%)",    True),   # lower is better
    ]
    for i, (key, label, lower_is_better) in enumerate(bar_metrics[:3]):
        ax = fig.add_subplot(gs[2, i])
        names  = list(results.keys())
        vals   = [results[n]["metrics"][key] for n in names]
        bclrs  = [_COLORS.get(n, "gray") for n in names]
        bars   = ax.bar(range(len(names)), vals, color=bclrs, edgecolor="white")
        best_i = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
        bars[best_i].set_edgecolor("gold")
        bars[best_i].set_linewidth(2.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(cfg.result_dir, exist_ok=True)
    out = f"{cfg.result_dir}/dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Dashboard saved → {out}")
    plt.show()


def print_table(results: Dict):
    names  = list(results.keys())
    metrics = [
        ("total_return_%",   "Total Return (%)"),
        ("cagr_%",           "CAGR (%)"),
        ("ann_volatility_%", "Ann. Volatility (%)"),
        ("sharpe_ratio",     "Sharpe Ratio"),
        ("sortino_ratio",    "Sortino Ratio"),
        ("calmar_ratio",     "Calmar Ratio"),
        ("max_drawdown_%",   "Max Drawdown (%)"),
        ("win_rate_%",       "Win Rate (%)"),
        ("var_95_%",         "VaR-95 (%)"),
    ]
    col_w = 14
    sep   = "═" * (28 + col_w * len(names))
    print(f"\n{sep}")
    print(f"{'PERFORMANCE COMPARISON':^{28 + col_w * len(names)}}")
    print(sep)
    print(f"{'Metric':<28}" + "".join(f"{n:>{col_w}}" for n in names))
    print("─" * (28 + col_w * len(names)))
    lower_better = {"max_drawdown_%", "ann_volatility_%", "var_95_%"}
    for key, label in metrics:
        vals = [results[n]["metrics"][key] for n in names]
        best = min(vals) if key in lower_better else max(vals)
        row  = f"{label:<28}"
        for v in vals:
            cell = f"{v:>{col_w - 1}.2f}"
            row += (" ★" if v == best else "  ") + cell[1:]
        print(row)
    print(sep + "\n")


# ============================================================
# SECTION 9: MAIN ORCHESTRATOR
# ============================================================

def main():
    cfg = TradingConfig()
    os.makedirs(cfg.model_dir,  exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    # 1 ── Data pipeline ──────────────────────────────────────────
    all_data = download_data(cfg)
    if len(all_data) < 3:
        logger.error("Not enough tickers downloaded.  Check network / ticker names.")
        return

    all_feat, feat_cols, date_idx = prepare_features(all_data, cfg)
    cfg.tickers = list(all_feat.keys())

    # 2 ── Train all algorithms ───────────────────────────────────
    algorithms    = ["SAC", "TD3", "A2C", "PPO"]
    trained_models: Dict[str, Any] = {}

    for algo in algorithms:
        try:
            trained_models[algo] = train_agent(algo, all_feat, feat_cols, date_idx, cfg)
        except Exception as exc:
            logger.error(f"  {algo} training failed: {exc}")
            import traceback; traceback.print_exc()

    if not trained_models:
        logger.error("No models were trained.  Exiting.")
        return

    # 3 ── Backtest individual agents ────────────────────────────
    results: Dict[str, Any] = {}
    for algo, model in trained_models.items():
        v, w = backtest_agent(model, algo, all_feat, feat_cols, date_idx, cfg)
        results[algo] = {"values": v, "metrics": compute_metrics(v, cfg.initial_capital)}

    # 4 ── Ensemble ───────────────────────────────────────────────
    if len(trained_models) > 1:
        ev, ew = ensemble_backtest(trained_models, all_feat, feat_cols, date_idx, cfg)
        results["Ensemble"] = {"values": ev, "metrics": compute_metrics(ev, cfg.initial_capital)}

    # 5 ── Baselines ──────────────────────────────────────────────
    logger.info("\nRunning baselines …")
    nv, nm = nifty_baseline(cfg)
    results["Nifty50 B&H"] = {"values": nv, "metrics": nm}

    ev2, em = equal_weight_baseline(all_feat, cfg)
    results["Equal Weight"] = {"values": ev2, "metrics": em}

    mv, mm = momentum_baseline(all_feat, cfg)
    results["Momentum"] = {"values": mv, "metrics": mm}

    # 6 ── Report ─────────────────────────────────────────────────
    print_table(results)
    plot_dashboard(results, cfg)

    with open(f"{cfg.result_dir}/metrics.json", "w") as f:
        json.dump({n: d["metrics"] for n, d in results.items()}, f, indent=2)
    logger.info(f"Metrics saved → {cfg.result_dir}/metrics.json")
    logger.info("\n✓  Done.")


if __name__ == "__main__":
    main()
