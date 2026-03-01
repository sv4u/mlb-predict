# MLB Win Probability

Research-grade pre-game win probability model for MLB regular season games, 2000–2025.

## Model performance (v3 — out-of-sample, expanding-window CV)

| Model               | Mean Brier | Mean Accuracy | Cal. Error | Best season |
| ------------------- | ---------- | ------------- | ---------- | ----------- |
| Logistic regression | 0.2443     | 56.2%         | 0.030      | 2019: 58.4% |
| LightGBM (Optuna)   | 0.2448     | 55.9%         | 0.029      | 2019: 58.7% |
| XGBoost (Optuna)    | **0.2442** | 56.4%         | 0.029      | 2019: 59.2% |
| Stacked ensemble    | **0.2441** | 56.3%         | 0.029      | 2019: 59.1% |

All metrics are fully out-of-sample: train on seasons < N, evaluate on season N.
The stacked ensemble is the default production model.

---

## Models

The system trains four models on 66 pre-game features using an **expanding-window protocol** — each season N is evaluated using a model trained exclusively on seasons before N, so all reported metrics are fully out-of-sample. Every model goes through **Platt calibration** (a sigmoid meta-layer fitted on a held-out calibration set) and **time-weighted training** (exponential decay rate 0.12 per season, so 2024 weight = 1.0, 2020 weight ≈ 0.61, 2015 weight ≈ 0.30).

### Logistic Regression

A regularised linear model that serves as the interpretable baseline. All 66 features are z-score standardised before fitting. Because the decision boundary is a hyperplane, the model captures additive effects — for example, a larger Elo differential increases home-win probability by a fixed amount regardless of the other features. Its simplicity makes it fast, stable, and easy to audit.

- **Regularisation**: L2 (ridge), `C=1.0`
- **Solver**: L-BFGS with up to 1 000 iterations
- **Interpretability**: SHAP attributions are computed directly from `coef × z-score` — no approximate explainer needed
- **When to use**: Speed-critical inference, auditing individual predictions, baseline comparison

### LightGBM

Microsoft's LightGBM grows an ensemble of shallow decision trees in sequence, where each tree corrects the residual errors of the ones before it. Unlike logistic regression, it captures **non-linear interactions** — for example, a high Elo differential combined with a good home/away split may carry a larger joint effect than either feature alone.

- **Hyperparameters**: 60-trial Optuna Bayesian search minimising out-of-sample Brier score (typical result: `num_leaves≈63`, `learning_rate≈0.05`, `n_estimators≈500`)
- **Interpretability**: Tree-based SHAP values via `shap.TreeExplainer`
- **When to use**: Fast batch inference at scale; often competitive with XGBoost

### XGBoost

DMLC's XGBoost is the other dominant gradient-boosted tree library. Its regularisation scheme (`min_child_weight`, separate L1/L2 penalties on leaf weights) and histogram-based split finding produce probability estimates that are **complementary** to LightGBM — they tend to disagree most on uncertain games near 50%, which makes them useful ensemble partners. XGBoost typically achieves the best **single-model** Brier score in this system.

- **Hyperparameters**: 60-trial Optuna Bayesian search (typical result: `max_depth≈6`, `learning_rate≈0.05`, `n_estimators≈500`)
- **Interpretability**: Tree-based SHAP values via `shap.TreeExplainer`
- **When to use**: Highest standalone accuracy; default choice when not ensembling

### Stacked Ensemble (default production model)

The stacked ensemble never sees raw features. Instead, it takes the **calibrated probability outputs** of all three base models as its three inputs and trains a logistic-regression **meta-learner** to find the optimal blend. Because each base model makes different errors, the meta-learner learns to up-weight whichever model is most confident in each probability range.

```
 Logistic prob  ─┐
 LightGBM prob  ─┼──▶  Logistic meta-learner  ──▶  P(home win)
 XGBoost prob   ─┘
```

- **Meta-learner**: `LogisticRegression(C=0.5)` — slight regularisation prevents over-fitting to the calibration set
- **Training**: The meta-learner is fit on the same held-out calibration split used for Platt scaling, so base-model probabilities are out-of-sample relative to the meta-learner
- **When to use**: Always — this is the default and achieves the best Brier score and calibration

### Training techniques

| Technique | What it does |
| --- | --- |
| **Platt calibration** | Fits a sigmoid on the base model's log-odds output using a held-out calibration set. Corrects systematic over- or under-confidence so that predicted 65% games actually win ~65% of the time. |
| **Time-weighted training** | Exponential decay (`rate=0.12` per season) gives recent seasons more influence. This adapts the model to baseball rule changes — the 2023 shift ban, pitch clock, and larger bases shift team-level stats in ways that older seasons do not reflect. |
| **Optuna HPO** | Bayesian hyperparameter search (60 trials per model type) over a 3-season expanding-window objective. Searches `learning_rate`, `num_leaves`/`max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, L1/L2 regularisation. |
| **Expanding-window CV** | For evaluation season N, the model is trained on all seasons before N. No future data ever leaks into training or calibration. |

---

## Features (66 total)

### Team performance

- **Elo rating** (home, away, diff) — sequential cross-season rating with regression-to-mean at each season start; accounts for opponent quality
- **Multi-window rolling** (15 / 30 / 60 games, cross-season warm-start): win%, run differential, Pythagorean expectation
- **EWMA rolling** (span=20): exponentially-weighted recent-form metrics
- **Home/away performance splits**: team win% and Pythagorean computed separately in home games vs. road games

### Context & fatigue

- **Streak**: current win (+) or loss (−) streak for each team
- **Rest days**: calendar days since last game (capped at 10)
- **Season progress**: 0 = opener, 1 = final day

### Pitcher quality

- **Prior-season SP ERA, K/9, BB/9** from the MLB Stats API — one row per pitcher per season, joined by name

### Advanced team metrics (FanGraphs, prior season)

- **Batting**: wOBA, Barrel%, Hard Hit%
- **Pitching**: FIP, xFIP, K%

### Park

- **Park run factor** — historical runs per game at the venue vs. league average

### Differential features

- Pythagorean diff, EWMA Pythagorean diff, home/road split diff, SP ERA diff, wOBA diff, FIP diff

---

## Quick start

### Install

```bash
git clone <repo>
cd mlb-winprob
pip install -e .
```

### Full data ingestion

```bash
# 1. Fetch MLB schedules (2000–2025)
python scripts/ingest_schedule.py --seasons $(seq 2000 2025)

# 2. Fetch Retrosheet gamelogs
python scripts/ingest_gamelogs.py --seasons $(seq 2000 2025)

# 3. Build Retrosheet ↔ MLB crosswalk
python scripts/build_crosswalk.py --seasons $(seq 2000 2025)

# 4. Fetch individual pitcher season stats
python scripts/ingest_pitcher_stats.py --seasons $(seq 2000 2025)

# 5. Fetch FanGraphs team advanced metrics
python scripts/ingest_fangraphs.py --seasons $(seq 2002 2025)
```

### Build features

```bash
python scripts/build_features.py --seasons $(seq 2000 2025)
```

### Train models (with Optuna HPO)

```bash
python scripts/train_model.py --hpo --hpo-trials 60
```

Skip HPO if you just want to re-train with existing hyperparameters:

```bash
python scripts/train_model.py
```

### Launch the web dashboard

```bash
python scripts/serve.py
# Open http://localhost:8087
```

### CLI query tool

```bash
# Game detail with SHAP attribution
python scripts/query_game.py --game-pk 745444

# Dodgers vs. Padres on opening day 2024
python scripts/query_game.py --home SDP --away LAD --season 2024 --date 2024-03-20

# All 2024 Dodgers home games (compact)
python scripts/query_game.py --home LAD --season 2024 --show-schedule

# Biggest upsets of 2024
python scripts/query_game.py --season 2024 --show-upsets --top-n 10

# Brief one-line output
python scripts/query_game.py --home NYY --season 2025 --brief
```

---

## Data pipeline

```
MLB Stats API        Retrosheet gamelogs      FanGraphs
      │                     │                     │
      ▼                     ▼                     ▼
 schedules/          retrosheet/              fangraphs/          pitcher_stats/
 games_YYYY.parquet  gamelogs_YYYY.parquet    fangraphs_YYYY.parquet  pitchers_YYYY.parquet
      │                     │
      └──── crosswalk ───────┘
            game_id_map_YYYY.parquet
                    │
                    ▼
              features/
         features_YYYY.parquet   ←── 66 features per game
                    │
                    ▼
               models/
     logistic_v3_train2025/       lightgbm_v3_train2025/
     xgboost_v3_train2025/        hpo_lightgbm.json  hpo_xgboost.json
     cv_summary_v3.json
```

---

## Data locations

| Path                            | Contents                                                   |
| ------------------------------- | ---------------------------------------------------------- |
| `data/raw/schedules/`           | Raw MLB Stats API JSON responses                           |
| `data/raw/gamelogs/`            | Raw Retrosheet TXT game logs                               |
| `data/processed/schedules/`     | `games_YYYY.parquet` + checksums                           |
| `data/processed/retrosheet/`    | `gamelogs_YYYY.parquet`                                    |
| `data/processed/crosswalk/`     | `game_id_map_YYYY.parquet`, coverage report                |
| `data/processed/pitcher_stats/` | `pitchers_YYYY.parquet` (MLB API individual stats)         |
| `data/processed/fangraphs/`     | `fangraphs_YYYY.parquet` (FanGraphs team advanced metrics) |
| `data/processed/features/`      | `features_YYYY.parquet` (66-feature matrix)                |
| `data/models/`                  | Trained model artifacts + HPO results + CV summaries       |
| `data/processed/predictions/`   | Immutable prediction snapshots                             |
| `data/processed/drift/`         | Drift monitoring logs                                      |

---

## Querying predictions (Python)

```python
import pandas as pd
from pathlib import Path

# Load all predictions
frames = [pd.read_parquet(f) for f in sorted(Path("data/processed/features").glob("features_*.parquet"))]
df = pd.concat(frames, ignore_index=True)

from winprob.model.artifacts import latest_artifact, load_model
from winprob.model.train import _predict_proba

model, meta = load_model(latest_artifact("logistic", version="v3"))
df["prob"] = _predict_proba(model, df[meta.feature_cols].fillna(0.5))

# 2024 games with high home-team probability
df24 = df[df["season"] == 2024].sort_values("prob", ascending=False)
print(df24[["date","home_retro","away_retro","prob","home_win"]].head(10))

# Accuracy by favourite probability bucket
df24["fav_won"] = ((df24["prob"] >= 0.5) == (df24["home_win"] == 1)).astype(float)
print(df24.groupby(pd.cut(df24["prob"].clip(0.5,0.99), 5))["fav_won"].mean())
```

---

## Web dashboard

Start the dashboard with `python scripts/serve.py`, then open `http://localhost:8087`.

Features:

- **Games browser** — filter by season, home team, away team, or date; sortable columns
- **Game detail** — probability bars, SHAP factor attribution chart, key stats comparison
- **Biggest upsets** — all-time or by season, filtered by minimum favourite probability
- **CV accuracy chart** — out-of-sample accuracy trend across all 4 model types

### API endpoints

| Endpoint                                   | Description                                 |
| ------------------------------------------ | ------------------------------------------- |
| `GET /api/seasons`                         | List available seasons                      |
| `GET /api/teams`                           | List all teams                              |
| `GET /api/games?season=&home=&away=&date=` | Paginated game list with predictions        |
| `GET /api/games/{game_pk}`                 | Full detail + SHAP attribution for one game |
| `GET /api/upsets?season=&min_prob=0.65`    | Biggest upsets by favourite probability     |
| `GET /api/cv-summary`                      | Model CV results by season                  |

---

## Project structure

```
mlb-winprob/
├── src/winprob/
│   ├── mlbapi/          # MLB Stats API client (async, rate-limited)
│   │   ├── client.py
│   │   ├── schedule.py
│   │   └── pitcher_stats.py
│   ├── statcast/        # FanGraphs / Statcast advanced metrics
│   │   └── fangraphs.py
│   ├── features/        # Feature engineering pipeline
│   │   ├── elo.py       # Sequential Elo rating
│   │   ├── team_stats.py# Multi-window, EWMA, home/away splits
│   │   ├── pitcher_stats.py # Gamelog-based pitcher ERA
│   │   ├── park_factors.py
│   │   └── builder.py   # Assembles 66-feature matrix
│   ├── model/           # Model training and evaluation
│   │   ├── train.py     # LR + LightGBM + XGBoost + stacked, Optuna, time-weighted
│   │   ├── evaluate.py
│   │   └── artifacts.py # Save / load model artifacts
│   ├── predict/         # Prediction snapshots
│   │   └── snapshot.py
│   ├── drift/           # Drift monitoring
│   │   └── compute.py
│   └── app/             # FastAPI web dashboard
│       ├── main.py
│       ├── data_cache.py
│       └── templates/
│           ├── index.html
│           └── game.html
├── scripts/
│   ├── ingest_schedule.py
│   ├── ingest_gamelogs.py
│   ├── build_crosswalk.py
│   ├── ingest_pitcher_stats.py   # MLB Stats API individual pitcher stats
│   ├── ingest_fangraphs.py       # FanGraphs team advanced metrics
│   ├── build_features.py         # Build 66-feature matrices
│   ├── train_model.py            # Optuna HPO + expanding-window CV + production models
│   ├── run_predictions.py        # Snapshot predictions
│   ├── compute_drift.py          # Drift monitoring
│   ├── query_game.py             # Human-centric CLI query tool
│   └── serve.py                  # Launch FastAPI dashboard
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── schedules/
│   │   ├── retrosheet/
│   │   ├── crosswalk/
│   │   ├── pitcher_stats/
│   │   ├── fangraphs/
│   │   ├── features/
│   │   ├── predictions/
│   │   └── drift/
│   └── models/
├── pyproject.toml
└── README.md
```

---

## Attribution

Game log data from **Retrosheet** (retrosheet.org).

> The information used here was obtained free of charge from and is copyrighted by Retrosheet.
> Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.

Advanced metrics from **FanGraphs** (fangraphs.com) via the `pybaseball` library.
Schedule and player data from the **MLB Stats API** (statsapi.mlb.com).
