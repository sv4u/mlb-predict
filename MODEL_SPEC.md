# MODEL_SPEC.md
## MLB Win Probability Modeling System — Modeling Specification

This document defines the mathematical and engineering contracts for modeling pre-game
win probabilities across MLB regular seasons 2000–2026.

---

# 1. Problem Definition

For each scheduled MLB regular-season game `g`, estimate:

- `P(HomeWin | x_g)`

where `x_g ∈ ℝ^66` contains team- and player-level features constructed exclusively
from data available before first pitch.

Outputs are probabilities in `[0, 1]` evaluated by Brier score, calibration error,
and accuracy. All reported metrics are **fully out-of-sample** (expanding-window CV).

---

# 2. Feature Set (66 features)

Features must satisfy:

- derived only from data available at scoring time (no leakage)
- deterministic computation
- stable schema versioning (see `data/models/cv_summary_v3.json`)

## 2.1 Team strength (Elo)

| Feature | Description |
| --- | --- |
| `home_elo`, `away_elo` | Sequential cross-season Elo; K=20, HFA offset=+35 |
| `elo_diff` | `home_elo − away_elo` |

## 2.2 Multi-window rolling stats (15 / 30 / 60 games, cross-season warm-start)

| Feature group | Windows |
| --- | --- |
| Win percentage | 15, 30, 60 |
| Run differential | 15, 30, 60 |
| Pythagorean expectation | 15, 30, 60 |

Applied separately to home and away teams; differentials computed for each window.

## 2.3 EWMA rolling stats (span=20)

| Feature | Description |
| --- | --- |
| `home_win_ewm`, `away_win_ewm` | Exponentially weighted win percentage |
| `home_rd_ewm`, `away_rd_ewm` | Exponentially weighted run differential |
| `home_pythag_ewm`, `away_pythag_ewm` | Exponentially weighted Pythagorean expectation |
| `pythag_ewm_diff` | `home_pythag_ewm − away_pythag_ewm` |

## 2.4 Home / away performance splits

| Feature | Description |
| --- | --- |
| `home_win_pct_home_only` | Rolling win% in home games only (home team) |
| `away_win_pct_away_only` | Rolling win% in road games only (away team) |
| `home_pythag_home_only` | Rolling Pythagorean in home games (home team) |
| `away_pythag_away_only` | Rolling Pythagorean in road games (away team) |
| `home_away_split_diff` | `home_win_pct_home_only − away_win_pct_away_only` |
| `pythag_ha_diff` | `home_pythag_home_only − away_pythag_away_only` |

## 2.5 Context and fatigue

| Feature | Description |
| --- | --- |
| `home_streak`, `away_streak` | Current win (+) / loss (−) streak |
| `home_rest`, `away_rest` | Calendar days since last game (capped at 10) |
| `season_progress` | `(game_index / total_games)` — 0.0 = opener, 1.0 = final day |

## 2.6 Starting pitcher quality (prior season, MLB Stats API)

| Feature | Description |
| --- | --- |
| `home_sp_era`, `away_sp_era` | Starter ERA from prior season |
| `home_sp_k9`, `away_sp_k9` | Strikeouts per 9 innings |
| `home_sp_bb9`, `away_sp_bb9` | Walks per 9 innings |
| `sp_era_diff` | `away_sp_era − home_sp_era` (positive = home advantage) |

## 2.7 FanGraphs advanced team metrics (prior season, via `pybaseball`)

| Feature | Description |
| --- | --- |
| `home_woba`, `away_woba` | Weighted on-base average |
| `home_barrel_pct`, `away_barrel_pct` | Barrel % |
| `home_hard_pct`, `away_hard_pct` | Hard Hit % |
| `home_fip`, `away_fip` | Fielding Independent Pitching |
| `home_xfip`, `away_xfip` | Expected FIP |
| `home_k_pct`, `away_k_pct` | Team strikeout % |
| `woba_diff` | `home_woba − away_woba` |
| `fip_diff` | `away_fip − home_fip` (positive = home advantage) |

## 2.8 Park

| Feature | Description |
| --- | --- |
| `park_factor` | Median runs-per-game at venue vs. league average (from historical gamelogs) |

---

# 3. Models

## 3.1 Training protocol

All four models share the same protocol:

- **Expanding-window cross-validation**: train on seasons < N, evaluate on season N.
  No future data leaks into training or calibration.
- **Time-weighted sample weights**: exponential decay at `rate=0.12` per season
  (e.g. 2024 weight = 1.0, 2020 weight ≈ 0.61, 2015 weight ≈ 0.30). Adapts the
  model to rule changes (shift ban 2023, pitch clock 2023, etc.).
- **Platt calibration**: a sigmoid meta-layer `σ(a·logit + b)` fitted on a held-out
  calibration split. Ensures that predicted 65% games actually win ~65% of the time.
- **Model artifact versioning**: `v3` is the current production version.

## 3.2 Logistic Regression

Regularised linear model; serves as the interpretable baseline.

`p_g = σ(wᵀ z_g + b)` where `z_g = (x_g − μ) / σ` (z-score standardised)

- **Regularisation**: L2 (ridge), `C=1.0`
- **Solver**: L-BFGS, up to 1 000 iterations
- **SHAP**: computed directly from `coef × z-score` — no approximate explainer needed
- **Production use**: audit, interpretability, baseline comparison

## 3.3 LightGBM

Gradient-boosted trees; captures non-linear feature interactions.

- **Hyperparameters**: 60-trial Optuna Bayesian search minimising out-of-sample Brier score
  (typical result: `num_leaves≈63`, `learning_rate≈0.05`, `n_estimators≈500`)
- **Best HPO stored at**: `data/models/hpo_lightgbm.json`
- **SHAP**: `shap.TreeExplainer`
- **Production use**: fast batch inference; competitive with XGBoost

## 3.4 XGBoost

Gradient-boosted trees with separate L1/L2 leaf regularisation.

- **Hyperparameters**: 60-trial Optuna Bayesian search
  (typical result: `max_depth≈6`, `learning_rate≈0.05`, `n_estimators≈500`)
- **Best HPO stored at**: `data/models/hpo_xgboost.json`
- **SHAP**: `shap.TreeExplainer`
- **Production use**: best single-model Brier score; default when not ensembling

## 3.5 Stacked Ensemble (default production model)

The meta-learner never sees raw features; it receives the calibrated probability
outputs of the three base models and blends them optimally.

```
 Logistic prob  ─┐
 LightGBM prob  ─┼──▶  LogisticRegression(C=0.5)  ──▶  P(home win)
 XGBoost prob   ─┘
```

- **Meta-learner**: `LogisticRegression(C=0.5)` fit on the same held-out calibration
  set used for Platt scaling, so base-model probabilities are out-of-sample
- **Production use**: achieves the best overall Brier score and calibration

---

# 4. Evaluation

## 4.1 Metrics

| Metric | Description |
| --- | --- |
| **Brier score** | Mean squared probability error (lower = better; range [0, 1]) |
| **Accuracy** | % of games where the model's favourite won |
| **Calibration error** | Mean absolute deviation between predicted probability and observed win rate |

## 4.2 Current out-of-sample results (v3, expanding-window CV, 2001–2025)

| Model               | Mean Brier | Mean Accuracy | Cal. Error |
| ------------------- | ---------- | ------------- | ---------- |
| Logistic regression | 0.2443     | 56.2%         | 0.030      |
| LightGBM (Optuna)   | 0.2448     | 55.9%         | 0.029      |
| XGBoost (Optuna)    | **0.2442** | 56.4%         | 0.029      |
| Stacked ensemble    | **0.2441** | 56.3%         | 0.029      |

Full season-by-season CV results: `data/models/cv_summary_v3.json`

## 4.3 Splits

- Train: all seasons before N (expanding window)
- Calibration / meta-learner: held-out split of the training set (20%)
- Test: season N (fully out-of-sample)

---

# 5. Versioning and Reproducibility

## 5.1 Semantic versioning

- **major** (`v1 → v2 → v3`): feature schema changes
- **minor**: new features or calibration approach within the same schema
- **patch**: bug fixes without schema changes

## 5.2 Required hashes per snapshot

Each prediction snapshot Parquet includes:

| Hash field | Source |
| --- | --- |
| `model_version` | e.g. `xgboost_v3_train2025` |
| `schedule_hash` | SHA256 of the schedule Parquet |
| `feature_hash` | SHA256 of the feature Parquet |
| `git_commit` | Current HEAD commit SHA |

Hashes are computed from canonical, sorted inputs to ensure reproducibility.

---

# 6. Integrity Constraints

Agents MUST NOT:
- train on test-season outcomes when generating predictions for that season
- use 2026 game results to train or calibrate any model (no 2026 results exist yet)
- overwrite model artifacts without bumping the version suffix
- alter historical prediction snapshots (they are immutable)
- introduce nondeterministic randomness without recording the seed in metadata

---

# 7. Roadmap

| Item | Status |
| --- | --- |
| Logistic regression baseline | ✅ Implemented (v3) |
| LightGBM + Optuna HPO | ✅ Implemented (v3) |
| XGBoost + Optuna HPO | ✅ Implemented (v3) |
| Stacked ensemble | ✅ Implemented (v3) |
| Platt calibration | ✅ Implemented |
| Time-weighted training | ✅ Implemented |
| Pitcher stats features | ✅ Implemented |
| FanGraphs team metrics | ✅ Implemented |
| Park factors | ✅ Implemented |
| 2026 pre-season predictions | ✅ Implemented |
| Drift monitoring | ✅ Implemented |
| SHAP attributions | ✅ Implemented |
| Lineup Monte Carlo simulation | ⬜ Planned |
| Hierarchical team Bayesian priors | ⬜ Planned |
| Market comparison module | ⬜ Planned |
