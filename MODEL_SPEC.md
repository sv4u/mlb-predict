# MODEL_SPEC.md

## MLB Prediction System — Modeling Specification

This document defines the mathematical and engineering contracts for modeling pre-game
win probabilities across MLB regular seasons 2000–2026.

---

# 1. Problem Definition

For each scheduled MLB regular season and spring training game `g`, estimate:

- `P(HomeWin | x_g)`

where `x_g ∈ ℝ^136` contains team- and player-level features constructed exclusively
from data available before first pitch.

Outputs are probabilities in `[0, 1]` evaluated by Brier score, calibration error,
and accuracy. All reported metrics are **fully out-of-sample** (expanding-window CV).

---

# 2. Feature Set (136 features)

Features must satisfy:

- derived only from data available at scoring time (no leakage)
- deterministic computation
- stable schema versioning (see `data/models/cv_summary_v4.json`)

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
| `home_win_pct_ewm`, `away_win_pct_ewm` | Exponentially weighted win percentage |
| `home_run_diff_ewm`, `away_run_diff_ewm` | Exponentially weighted run differential |
| `home_pythag_ewm`, `away_pythag_ewm` | Exponentially weighted Pythagorean expectation |
| `pythag_diff_ewm` | `home_pythag_ewm − away_pythag_ewm` |

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
| `home_rest_days`, `away_rest_days` | Calendar days since last game (capped at 10) |
| `season_progress` | `(game_index / total_games)` — 0.0 = opener, 1.0 = final day |

## 2.6 Starting pitcher quality (prior season, MLB Stats API)

| Feature | Description |
| --- | --- |
| `home_sp_era`, `away_sp_era` | Starter ERA from prior season |
| `home_sp_k9`, `away_sp_k9` | Strikeouts per 9 innings |
| `home_sp_bb9`, `away_sp_bb9` | Walks per 9 innings |
| `sp_era_diff` | `away_sp_era − home_sp_era` (positive = home advantage) |

## 2.7 Starting Pitcher WHIP

| Feature | Description |
| --- | --- |
| `home_sp_whip`, `away_sp_whip` | Walks + hits per inning pitched (prior season) |

## 2.8 FanGraphs Advanced Team Metrics (24 features)

Prior-season team-level advanced metrics from FanGraphs via `pybaseball`.

**Batting (6 stats × 2 teams = 12 features):**

| Feature | Description |
| --- | --- |
| `home_bat_woba`, `away_bat_woba` | Weighted on-base average |
| `home_bat_barrel_pct`, `away_bat_barrel_pct` | Barrel % |
| `home_bat_hard_pct`, `away_bat_hard_pct` | Hard hit % |
| `home_bat_iso`, `away_bat_iso` | Isolated power |
| `home_bat_babip`, `away_bat_babip` | BABIP |
| `home_bat_xwoba`, `away_bat_xwoba` | Expected weighted on-base average |

**Pitching (6 stats × 2 teams = 12 features):**

| Feature | Description |
| --- | --- |
| `home_pit_fip`, `away_pit_fip` | Fielding Independent Pitching |
| `home_pit_xfip`, `away_pit_xfip` | Expected FIP |
| `home_pit_k_pct`, `away_pit_k_pct` | Team strikeout % |
| `home_pit_bb_pct`, `away_pit_bb_pct` | Team walk % |
| `home_pit_hr_fb`, `away_pit_hr_fb` | HR/FB ratio |
| `home_pit_whip`, `away_pit_whip` | WHIP (team-level FanGraphs) |

## 2.9 Statcast Individual Player Stats (6 features)

| Feature | Description |
| --- | --- |
| `home_lineup_xwoba`, `away_lineup_xwoba` | Lineup-weighted xwOBA (prior-season Statcast) |
| `home_lineup_barrel_pct`, `away_lineup_barrel_pct` | Lineup-weighted barrel% (prior-season) |
| `home_sp_est_woba`, `away_sp_est_woba` | Starting pitcher estimated xwOBA allowed |

## 2.10 Bullpen (8 features)

| Feature | Description |
| --- | --- |
| `home_bullpen_usage_15`, `home_bullpen_usage_30` | Bullpen IP per game (home, 15/30 window) |
| `away_bullpen_usage_15`, `away_bullpen_usage_30` | Bullpen IP per game (away, 15/30 window) |
| `home_bullpen_era_proxy_15`, `home_bullpen_era_proxy_30` | Bullpen ERA proxy (home, 15/30 window) |
| `away_bullpen_era_proxy_15`, `away_bullpen_era_proxy_30` | Bullpen ERA proxy (away, 15/30 window) |

## 2.11 Lineup Continuity (2 features)

| Feature | Description |
| --- | --- |
| `home_lineup_continuity` | Fraction of prior game's lineup retained (home) |
| `away_lineup_continuity` | Fraction of prior game's lineup retained (away) |

## 2.12 Run Distribution (4 features)

| Feature | Description |
| --- | --- |
| `home_run_std_30`, `away_run_std_30` | Scoring variance (30-game window) |
| `home_one_run_win_pct_30`, `away_one_run_win_pct_30` | Close-game win% (30-game window) |

## 2.13 Contextual Features (3 features)

| Feature | Description |
| --- | --- |
| `day_night` | 1 = day game, 0 = night game |
| `interleague` | 1 = interleague matchup |
| `day_of_week` | 0 (Monday) – 6 (Sunday), normalized |

## 2.14 Vegas Implied Probability (2 features)

| Feature | Description |
| --- | --- |
| `vegas_implied_home_win` | Implied home win probability from opening moneyline |
| `vegas_line_movement` | Change in implied probability (opening → closing) |

## 2.15 Weather (3 features)

| Feature | Description |
| --- | --- |
| `game_temp_f` | Game-time temperature (°F) |
| `game_wind_mph` | Wind speed (mph) |
| `game_humidity` | Relative humidity (%) |

## 2.16 Differential Features (9 team + 3 player = 12 features)

| Feature | Formula |
| --- | --- |
| `pythag_diff_30` | `home_pythag_30 − away_pythag_30` |
| `pythag_diff_ewm` | `home_pythag_ewm − away_pythag_ewm` |
| `home_away_split_diff` | `home_win_pct_home_only − away_win_pct_away_only` |
| `sp_era_diff` | `away_sp_era − home_sp_era` |
| `woba_diff` | `home_bat_woba − away_bat_woba` |
| `fip_diff` | `away_pit_fip − home_pit_fip` |
| `xwoba_diff` | `home_bat_xwoba − away_bat_xwoba` |
| `whip_diff` | `away_pit_whip − home_pit_whip` |
| `iso_diff` | `home_bat_iso − away_bat_iso` |
| `lineup_strength_diff` | `home_lineup_strength − away_lineup_strength` (Stage 1) |
| `sp_quality_diff` | `home_sp_quality − away_sp_quality` (Stage 1) |
| `matchup_advantage_diff` | `home_lineup_vs_sp − away_lineup_vs_sp` (Stage 1) |

## 2.17 Stage 1 Player Model Features (14 per-team + 3 differentials = 17 features)

Produced by the Stage 1 PyTorch player embedding model.  Per-player EWMA rolling stats, learned player ID embeddings, and biographical data are aggregated across the batting lineup using batting-order position weights.

| Feature | Description |
| --- | --- |
| `home_lineup_strength`, `away_lineup_strength` | Neural lineup quality score (0–1 scale) |
| `home_top3_quality`, `away_top3_quality` | Average quality of batters 1–3 |
| `home_bottom3_quality`, `away_bottom3_quality` | Average quality of batters 7–9 |
| `home_lineup_variance`, `away_lineup_variance` | Std dev of individual player quality |
| `home_platoon_advantage`, `away_platoon_advantage` | Learned platoon interaction vs opposing SP |
| `home_sp_quality`, `away_sp_quality` | Neural starting pitcher quality score |
| `home_lineup_vs_sp`, `away_lineup_vs_sp` | Matchup interaction: lineup vs opposing SP |

Differentials (`lineup_strength_diff`, `sp_quality_diff`, `matchup_advantage_diff`) are listed in §2.16.

---

# 3. Models

## 3.1 Training protocol

All six models share the same protocol:

- **Expanding-window cross-validation**: train on seasons < N, evaluate on season N.
  No future data leaks into training or calibration.
- **Time-weighted sample weights**: exponential decay at `rate=0.12` per season
  (e.g. 2024 weight = 1.0, 2020 weight ≈ 0.61, 2015 weight ≈ 0.30). Adapts the
  model to rule changes (shift ban 2023, pitch clock 2023, etc.).
- **Spring training weighting**: spring training games receive a reduced sample weight
  (default 0.25×) to contribute signal without dominating regular-season patterns.
- **Pre-training validation**: all seasons 2000–current must have schedule and
  regular-season feature files present before training.
- **Calibration**: isotonic calibration for tree models (LightGBM, XGBoost, CatBoost);
  Platt calibration (sigmoid meta-layer `σ(a·logit + b)`) for linear and neural models
  (logistic, MLP). Fitted on a held-out calibration split. Ensures that predicted 65%
  games actually win ~65% of the time.
- **Model artifact versioning**: `v4` is the current production version.

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

## 3.4 CatBoost

Yandex's CatBoost uses ordered boosting and symmetric (oblivious) decision trees.
Acts as a complementary tree model in the stacked ensemble.

- **Regularisation**: L2 leaf regularisation, learning rate decay
- **Architecture**: Symmetric (oblivious) trees with ordered boosting
- **SHAP**: `shap.TreeExplainer`

## 3.5 XGBoost

Gradient-boosted trees with separate L1/L2 leaf regularisation.

- **Hyperparameters**: 60-trial Optuna Bayesian search
  (typical result: `max_depth≈6`, `learning_rate≈0.05`, `n_estimators≈500`)
- **Best HPO stored at**: `data/models/hpo_xgboost.json`
- **SHAP**: `shap.TreeExplainer`
- **Production use**: best single-model Brier score; default when not ensembling

## 3.6 MLP (Neural Network)

Multi-layer perceptron with three hidden layers (128, 64, 32 units) and ReLU
activations. Features are z-score normalised before training.

- **Architecture**: 128 → 64 → 32 → 1, Adam optimiser
- **Regularisation**: L2 weight decay (alpha)
- **SHAP**: Not supported (coefficient ranking instead)

## 3.7 Stacked Ensemble (default production model)

The meta-learner never sees raw features; it receives the calibrated probability
outputs of the five base models and blends them optimally.

```
 Logistic prob  ─┐
 LightGBM prob  ─┤
 XGBoost prob   ─┼──▶  LogisticRegression(C=0.5)  ──▶  P(home win)
 CatBoost prob  ─┤
 MLP prob       ─┘
```

- **Meta-learner**: `LogisticRegression(C=0.5)` fit on the same held-out calibration
  set used for base-model calibration, so base-model probabilities are out-of-sample
- **Production use**: achieves the best overall Brier score and calibration

---

# 4. Evaluation

## 4.1 Metrics

| Metric | Description |
| --- | --- |
| **Brier score** | Mean squared probability error (lower = better; range [0, 1]) |
| **Accuracy** | % of games where the model's favourite won |
| **Calibration error** | Mean absolute deviation between predicted probability and observed win rate |

## 4.2 Current out-of-sample results

v4 results will be available after the next training run with Stage 1 player features.

Full season-by-season CV results: `data/models/cv_summary_v4.json`

## 4.3 Splits

- Train: all seasons before N (expanding window)
- Calibration / meta-learner: held-out split of the training set (20%)
- Test: season N (fully out-of-sample)

---

# 5. Versioning and Reproducibility

## 5.1 Semantic versioning

- **major** (`v1 → v2 → v3 → v4`): feature schema changes
- **minor**: new features or calibration approach within the same schema
- **patch**: bug fixes without schema changes

## 5.2 Required hashes per snapshot

Each prediction snapshot Parquet includes:

| Hash field | Source |
| --- | --- |
| `model_version` | e.g. `xgboost_v4_train2025` |
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
| CatBoost + Optuna HPO | ✅ Implemented (v3) |
| MLP neural network | ✅ Implemented (v3) |
| Spring training data integration | ✅ Implemented |
| Stage 1 player embedding model (v4) | ✅ Implemented |
| Per-pitcher game logs (MLB Stats API, v4) | ✅ Implemented |
| Feature schema 119 → 136 (v4) | ✅ Implemented |
| Pre-training data validation | ✅ Implemented |
| Drift monitoring | ✅ Implemented |
| SHAP attributions | ✅ Implemented |
| Lineup Monte Carlo simulation | ⬜ Planned |
| Hierarchical team Bayesian priors | ⬜ Planned |
| Market comparison module | ⬜ Planned |
