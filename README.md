# Enhanced RL Trading System for NSE Indian Equities — v2.0

> **"High returns with acceptable risk"** — a disciplined improvement over the original PPO system that suffered from extreme drawdowns (~78%) and poor risk-adjusted performance.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                │
│  yfinance download → OHLCV for 10 NSE large-caps + Nifty 50    │
│  compute_ticker_features() → 26 features per stock per day     │
│  add_cross_sectional_features() → 3 relative rank features     │
│  Align to common dates, standardise, clip to [-3, 3]           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  NsePortfolioEnv (gymnasium)                    │
│  Action:  ℝ^N (logits) → softmax → capped portfolio weights   │
│  Obs:     features (flat) + current weights + port. state      │
│  Reward:  λ·log_ret − λ_dd·Δdrawdown − λ_to·turnover          │
│           + bounded Sortino bonus every 20 steps               │
│  Safety:  max 35% per stock, min 2% cash, blowup termination   │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
    │   SAC   │         │   TD3   │         │ A2C/PPO │
    │(primary)│         │(stable) │         │(baseline│
    └────┬────┘         └────┬────┘         └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Ensemble Agent │
                    │  (avg logits)   │
                    └────────┬────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   BACKTESTING & METRICS                         │
│  Test period: 2023-01-01 → 2024-12-31                          │
│  Metrics: Sharpe, Sortino, Calmar, MaxDD, VaR-95, Win Rate     │
│  Baselines: Nifty 50 B&H, Equal Weight, Momentum              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Improvements — In Detail

### 2.1  Continuous Portfolio Weight Actions

**Baseline** used discrete actions: *buy / sell / hold* per stock. This is a fundamental mismatch — portfolio management is a continuous problem. With 10 stocks and 3 actions each, the agent could express only 3^10 = 59,049 distinct portfolios, most of which are extreme all-in bets.

**v2.0** uses a continuous action vector in `[-1, 1]^N`. A softmax transforms these logits into a valid probability distribution (weights summing to 1). This gives the agent an infinite action resolution and naturally produces diversified positions.

```
raw action:       [-0.2,  1.1,  0.3, -0.8,  0.6, ...]   (logits from policy)
    ↓  softmax
weights:          [ 0.08, 0.24, 0.11, 0.05, 0.18, ...]   (non-neg, sum ≤ 0.98)
    ↓  cap at 35%
final weights:    [ 0.08, 0.24, 0.11, 0.05, 0.18, ...]   (enforced risk limits)
```

### 2.2  Risk-Adjusted Reward Function

The original reward was simply `portfolio_return`, which incentivises reckless leverage and volatility. We restructure it as:

```
R_t = λ_ret · log(V_t / V_{t-1})
    − λ_dd  · max(0, DD_t − DD_{t-1})        ← punish drawdown increases
    − λ_to  · Σ|Δw_i|                         ← penalise excessive trading
    + clip(Sortino_{20d}, −1, 1)              ← every 20 steps
```

This means:
- The agent is only rewarded for return **relative to the cost of the drawdown** needed to get there.
- Each new drawdown increment is explicitly penalised — not just the level.
- Excessive rebalancing is penalised, naturally reducing transaction costs.

### 2.3  Validation-Driven Checkpointing (by Calmar Ratio)

Instead of saving the model at the end of training (which is often overfit), we run a full validation episode every 20,000 steps and save only if the **Calmar ratio improves**. The Calmar ratio = CAGR / Max Drawdown, which is the exact metric we care about improving vs the baseline.

### 2.4  Expanded & Richer Feature Set

26 features per stock (vs ~8 in the baseline), organised into families:

- **Price return features**: 1d, 5d, 20d, log-return
- **Trend**: EMA-9/21 cross, EMA-21/55 cross, MACD, MACD histogram
- **Momentum**: RSI-14, RSI-28, ROC-10, ROC-20
- **Volatility**: rolling std (10/20/60d), vol regime ratio, Bollinger %B, BB width, ATR-14
- **Volume**: 20d volume ratio, OBV z-score
- **Regime**: distance from 52-week high/low, Hurst proxy, volatility percentile

Plus 3 cross-sectional rank features computed across all stocks simultaneously (see Section 5).

### 2.5  Ensemble Inference

After training SAC, TD3, A2C, and PPO independently, backtest averaging their action logits before applying the softmax. This reduces per-model variance significantly and tends to produce more stable, consistently positive performance.

---

## 3. Algorithm Comparison & Rationale

| Algorithm | Type | Key Property | Fit for Portfolio Trading |
|---|---|---|---|
| **SAC** | Off-policy, stochastic | Entropy regularisation prevents premature convergence to greedy, overfit policies | Primary agent |
| **TD3** | Off-policy, deterministic | Twin Q-networks reduce overestimation bias; OU noise for smooth action exploration | Strong alternative |
| **A2C** | On-policy, stochastic | Memory-efficient; entropy bonus prevents mode collapse | Useful in ensemble |
| **PPO** | On-policy, clipped | Stable baseline; easy to tune; matches original system for fair comparison | Comparison only |

**Why SAC is primary:** Portfolio weight allocation is a continuous control problem. SAC's maximum-entropy framework explicitly discourages policies that put all weight in one stock — it naturally encourages diversification without needing to hard-code it. In noisy financial environments, this translates to more robust out-of-sample performance.

**Why TD3 over DDPG:** DDPG is notoriously unstable in financial environments due to Q-value overestimation. TD3's twin critics and delayed policy updates fix both issues.

---

## 4. Feature Engineering

### Per-ticker features (26 total)

```
Category        Feature          Description
─────────────────────────────────────────────────────
Returns         log_ret          Daily log return
                ret_1d/5d/20d    Simple returns at multiple horizons

Trend           ema_9_21         (EMA9 - EMA21) / Close
                ema_21_55        (EMA21 - EMA55) / Close
                macd             MACD line / Close
                macd_hist        MACD histogram / Close

Momentum        rsi_14/28        RSI normalised to [-1, 1]
                roc_10/20        Rate of change

Volatility      vol_10/20/60     Rolling log-return std
                vol_ratio        vol_10 / vol_60  (regime signal)
                bb_pct           Bollinger %B  ≈ position in band
                bb_width         Bollinger band width  (vol proxy)
                atr_14           ATR / Close  (normalised)

Volume          vol_ratio_20     Volume / 20d average volume
                obv_z            OBV rolling z-score

Regime          dist_52h         Distance from 52-week high (≤ 0)
                dist_52l         Distance from 52-week low  (≥ 0)
                hurst_px         Simplified Hurst proxy
                vol_pct          Vol percentile in 60-day window
```

### Cross-sectional features (3 total, computed across all stocks)

```
cs_mom_rank    Percentile rank of 20d return vs all stocks (0=worst, 1=best)
cs_vol_rank    Percentile rank of low realised vol (0=most volatile, 1=calmest)
cs_rsi_rank    Percentile rank of RSI-14 (0=most oversold, 1=most overbought)
```

Cross-sectional features are critical because they tell the agent **which stocks are strong or weak relative to each other right now**, not just in absolute terms. This allows the agent to implement cross-sectional momentum (rotate into recent winners) — one of the most robust alpha signals in equity markets.

All features are standardised to zero mean, unit variance over the training period, and clipped to [-3, 3] to prevent extreme values from destabilising training.

---

## 5. Risk Management Framework

Risk controls operate at **two levels**:

### 5.1  Environment-level hard constraints

```python
# Applied every step, regardless of what the policy outputs:
max_position_fraction = 0.35   # No single stock > 35% of portfolio
min_cash_fraction     = 0.02   # Always keep ≥ 2% uninvested
blowup_threshold      = 0.20   # Episode terminates if portfolio < 20% of initial
```

The max position constraint is applied **after** the softmax, by clamping and re-normalising. This is equivalent to a hard stop that the agent cannot circumvent.

### 5.2  Reward-level soft constraints

```python
drawdown_penalty = 2.0   # Each 1% new drawdown costs 2× the reward from returns
turnover_penalty = 0.5   # Each unit of portfolio turnover subtracts from reward
```

These teach the agent to *prefer* controlled drawdowns and low-turnover strategies — not just avoid them mechanically.

### 5.3  Volatility-aware position sizing (implicit)

Because the agent's features include `vol_20`, `vol_60`, `vol_ratio`, and `atr_14`, it has all the information needed to scale positions down when volatility rises — it learns to do this if it reduces drawdown and thus improves the Calmar-based reward.

---

## 6. Reward Function Design

The full reward at each step is:

```
R_t = 100 · log(V_t / V_{t-1})
    − 2.0  · max(0, DD_t − DD_{t−1}) · 100
    − 0.5  · Σᵢ |w_i,t − w_i,t−1|
    + clip(Sortino(last 20 returns), −1, +1)    [every 20 steps]
```

**Scaling:** The base log-return is multiplied by 100 to bring it into a numerically suitable range for the neural network's value function. All penalty coefficients are on the same scale.

**Drawdown penalty design:** We penalise *increases* in drawdown (not the level). This is intentional — if the portfolio is already in drawdown but recovering, we don't want to penalise it further. We only penalise new deterioration.

**Sortino bonus:** Every 20 steps we compute the Sortino ratio over the most recent 20 returns and add a bounded version of it to the reward. This directly incentivises the agent to seek high returns relative to downside risk — exactly what the Calmar ratio measures at a macro level.

---

## 7. Walk-Forward Validation

The project uses a **hold-out validation set** (2022) between the training period (2018–2021) and the test period (2023–2024). This prevents a subtle but common bug: choosing training hyperparameters based on test set performance, which inflates expected results.

For production use, we recommend extending this to a full **walk-forward** scheme:

```
Fold 1: Train 2018–2019 → Val 2020 → Test 2020 H2
Fold 2: Train 2018–2020 → Val 2021 → Test 2021 H2
Fold 3: Train 2018–2021 → Val 2022 → Test 2023
Final:  Train 2018–2022 → Val 2022 → Deploy 2024
```

The current `RiskAdjustedCallback` already implements per-step validation tracking. To run multi-fold walk-forward, instantiate `train_agent()` multiple times with different `train_start`/`train_end`/`val_start`/`val_end` arguments.

---

## 8. Ensemble Inference

At inference time, all trained agents observe the same market state and each outputs an action (portfolio weight logits). We average these logits before applying the softmax:

```python
avg_action = mean([sac_action, td3_action, a2c_action, ppo_action])
weights = softmax(avg_action)   # applied inside the environment
```

**Why average logits, not weights?** Averaging weights after softmax is biased towards flat allocations. Averaging logits before softmax preserves the relative conviction of each agent, while the softmax still produces a valid probability distribution.

In practice, ensemble agents tend to:
- Have lower maximum drawdown than any single agent
- Have lower Sharpe ratio than the *best* single agent
- Have higher Sharpe ratio than the *average* single agent

They also generalise better to unseen market conditions, which is what matters most.

---

## 9. Installation & Usage

### Prerequisites

```bash
# Python 3.10+ recommended
pip install yfinance gymnasium stable-baselines3 scipy matplotlib pandas numpy
```

### Quick start

```bash
# Clone / download the project, then:
python main.py
```

This will:
1. Download NSE data for 10 tickers (2018–2024)
2. Compute all features
3. Train SAC, TD3, A2C, and PPO (≈ 500k steps each, ~15–30 min on CPU)
4. Run backtests + ensemble on 2023–2024 test set
5. Compare against Nifty 50, Equal Weight, and Momentum baselines
6. Save `results/dashboard.png` and `results/metrics.json`

### Faster run (for testing)

Edit `TradingConfig` in `main.py`:
```python
total_timesteps: int = 50_000    # 10× faster, lower quality
check_freq:      int = 10_000
```

### GPU training

`stable-baselines3` uses PyTorch. If you have a CUDA-capable GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
The code will automatically use it — no changes needed.

---

## 10. Configuration Reference

All settings live in the `TradingConfig` dataclass at the top of `main.py`.

| Parameter | Default | Description |
|---|---|---|
| `tickers` | 10 NSE large-caps | Universe of stocks |
| `train_start` / `train_end` | 2018–2021 | Training period |
| `val_start` / `val_end` | 2022 | Validation period (used for checkpointing) |
| `test_start` / `test_end` | 2023–2024 | Held-out test period |
| `initial_capital` | ₹10,00,000 | Starting portfolio value |
| `max_position_fraction` | 0.35 | Hard cap per stock (35%) |
| `min_cash_fraction` | 0.02 | Minimum cash buffer (2%) |
| `blowup_threshold` | 0.20 | Episode ends if value < 20% of initial |
| `reward_scaling` | 100.0 | Multiplier on log-return component |
| `drawdown_penalty` | 2.0 | Coefficient on drawdown increase penalty |
| `turnover_penalty` | 0.5 | Coefficient on portfolio turnover penalty |
| `total_timesteps` | 500,000 | Training budget per algorithm |
| `check_freq` | 20,000 | How often to run validation episode |

**Tuning tips:**
- Increase `drawdown_penalty` (try 3.0–5.0) if the agent still takes on too much risk.
- Decrease `turnover_penalty` (try 0.1) if the agent is too passive and misses trends.
- Add more tickers to the universe to improve diversification.
- Increase `total_timesteps` to 1M–2M for better convergence (requires more time).

---

## 11. Interpreting Results

After training, you will see a table like this (illustrative numbers):

```
════════════════════════════════════════════════════════════════════
                      PERFORMANCE COMPARISON
════════════════════════════════════════════════════════════════════
Metric                    SAC      TD3      A2C    Ensemble  Nifty50
────────────────────────────────────────────────────────────────────
Total Return (%)        ★ 52.1    44.8     38.2      48.6     30.4
CAGR (%)                ★ 23.8    20.8     17.9      22.3     14.4
Ann. Volatility (%)       18.2    17.1   ★ 14.8      16.9     13.2
Sharpe Ratio            ★  1.31    1.22     1.21     1.28      0.65
Sortino Ratio           ★  1.84    1.71     1.62     1.78      0.88
Calmar Ratio            ★  1.89    1.64     1.52     1.72      1.32
Max Drawdown (%)         -12.6   -12.7   ★ -11.8    -13.0    -10.9
Win Rate (%)            ★ 54.2    53.8     53.1      54.0     52.3
════════════════════════════════════════════════════════════════════
```

Key things to look for:

1. **Calmar ratio > 1.32** (Nifty baseline) means we've beaten the benchmark on risk-adjusted terms — this is the primary goal.
2. **Max drawdown < 30%** (far less than the baseline's 78%) confirms the risk controls are working.
3. **The Ensemble should be roughly in the middle** on all metrics — lower peak return than SAC, but lower max drawdown too.

If you see a Calmar ratio still below the Nifty benchmark, try increasing `drawdown_penalty` to 3.0 or 4.0.

---

## 12. Next Steps & Research Directions

### Short-term improvements
- **Hyperparameter optimisation**: Use Optuna to tune `drawdown_penalty`, `turnover_penalty`, network architecture, and learning rates jointly.
- **Walk-forward backtesting**: Run the full multi-fold scheme described in Section 8 to get statistically robust performance estimates.
- **Transaction cost model**: Add impact scaling based on ADV (average daily volume) — especially important for larger capital.

### Medium-term improvements
- **LSTM / Transformer policy network**: Replace the MLP with a sequence model that can explicitly learn temporal patterns. SB3 supports custom policy architectures via `policy_kwargs`.
- **Multi-objective optimisation**: Use a Pareto-optimal reward that explicitly trades off Sharpe vs max drawdown, rather than a weighted sum.
- **Regime-conditional models**: Train separate agents for bull/bear/sideways regimes (detected with HMM or k-means on volatility/trend features), and switch between them at inference time.

### Advanced directions
- **Options hedging**: Extend the action space to include Nifty options for hedging drawdowns, with realistic NSE F&O pricing.
- **Fundamental data integration**: Add quarterly earnings, P/E, and debt-to-equity as slow-moving features alongside the technical indicators.
- **Market microstructure**: Model intraday execution with VWAP/TWAP scheduling for larger position sizes.

---

## License

MIT License — use freely for research and personal projects. Not financial advice.

---

*Built with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [Gymnasium](https://gymnasium.farama.org/), [yfinance](https://pypi.org/project/yfinance/), and NumPy/Pandas.*
