"""Knowledge base for tools: feature descriptions, model docs, sabermetric glossary.

Sourced from scripts/query_game.py _FEATURE_LABELS and the technical wiki.
Queried on demand by the describe_feature tool and by MCP/documentation.
"""

from __future__ import annotations

# Feature name → short description (keep in sync with scripts/query_game.py _FEATURE_LABELS)
FEATURE_LABELS: dict[str, str] = {
    "home_elo": "Home team Elo rating",
    "away_elo": "Away team Elo rating",
    "elo_diff": "Elo advantage (home − away)",
    "home_win_pct_30": "Home team 30-game win%",
    "away_win_pct_30": "Away team 30-game win%",
    "home_pythag_30": "Home team 30-game Pythagorean",
    "away_pythag_30": "Away team 30-game Pythagorean",
    "home_win_pct_ewm": "Home team recent form (EWMA)",
    "away_win_pct_ewm": "Away team recent form (EWMA)",
    "home_pythag_ewm": "Home team recent Pythagorean (EWMA)",
    "away_pythag_ewm": "Away team recent Pythagorean (EWMA)",
    "home_win_pct_home_only": "Home team at-home win%",
    "away_win_pct_away_only": "Away team on-road win%",
    "home_sp_era": "Home starter ERA (prior season)",
    "away_sp_era": "Away starter ERA (prior season)",
    "home_sp_k9": "Home starter K/9 (prior season)",
    "away_sp_k9": "Away starter K/9 (prior season)",
    "home_bat_woba": "Home team wOBA (prior season)",
    "away_bat_woba": "Away team wOBA (prior season)",
    "home_pit_fip": "Home team FIP (prior season)",
    "away_pit_fip": "Away team FIP (prior season)",
    "pythag_diff_30": "Pythagorean edge (home − away, 30g)",
    "pythag_diff_ewm": "Recent Pythagorean edge (EWMA)",
    "home_away_split_diff": "Home/road performance edge",
    "sp_era_diff": "Pitcher ERA edge (lower = home advantage)",
    "woba_diff": "Batting quality edge (wOBA)",
    "fip_diff": "Pitching quality edge (FIP)",
    "park_run_factor": "Park run factor (1.0 = neutral)",
    "season_progress": "Point in season (0=opening, 1=end)",
    "home_streak": "Home team win streak (+) / loss streak (−)",
    "away_streak": "Away team win streak (+) / loss streak (−)",
    "home_rest_days": "Home team rest days",
    "away_rest_days": "Away team rest days",
    # Stage 1 player model features (v4)
    "home_lineup_strength": "Home lineup overall quality (Stage 1 neural model)",
    "away_lineup_strength": "Away lineup overall quality (Stage 1 neural model)",
    "home_top3_quality": "Home top-of-order batter quality (1–3 hitters)",
    "away_top3_quality": "Away top-of-order batter quality (1–3 hitters)",
    "home_bottom3_quality": "Home bottom-of-order batter quality (7–9 hitters)",
    "away_bottom3_quality": "Away bottom-of-order batter quality (7–9 hitters)",
    "home_lineup_variance": "Home lineup consistency (lower = more balanced)",
    "away_lineup_variance": "Away lineup consistency (lower = more balanced)",
    "home_platoon_advantage": "Home batters vs. away SP handedness advantage",
    "away_platoon_advantage": "Away batters vs. home SP handedness advantage",
    "home_sp_quality": "Home starting pitcher quality (Stage 1 neural model)",
    "away_sp_quality": "Away starting pitcher quality (Stage 1 neural model)",
    "home_lineup_vs_sp": "Home lineup matchup advantage vs. away SP",
    "away_lineup_vs_sp": "Away lineup matchup advantage vs. home SP",
    "lineup_strength_diff": "Lineup quality edge (home − away)",
    "sp_quality_diff": "Pitcher quality edge (home − away)",
    "matchup_advantage_diff": "Matchup advantage edge (home − away)",
}

# Sabermetric / stat terms (from wiki)
GLOSSARY: dict[str, str] = {
    "elo": "Elo rating: chess-style team strength. Starts at 1500; winner gains points, loser drops. Home gets +24. Positive elo_diff favors home.",
    "pythagorean": "Pythagorean expectation: expected win% from runs scored and allowed, RS²/(RS²+RA²). Teams above it are often 'lucky'.",
    "ewma": "Exponentially weighted moving average: recent games weighted more than older ones. Captures hot/cold streaks.",
    "woba": "Weighted On-Base Average: overall batting value per PA. Better than AVG because it weights hits by run value.",
    "fip": "Field Independent Pitching: ERA-like measure using only K, BB, HR. Strips out defense and batted-ball luck.",
    "era": "Earned Run Average: earned runs per 9 innings. Lower is better for pitchers.",
    "k9": "Strikeouts per 9 innings. Higher usually means dominant stuff.",
    "park factor": "Runs at this ballpark vs league average. 1.0 = neutral; Coors ~1.3, pitcher parks ~0.85.",
    "shap": "SHAP values: each feature's contribution to the prediction. Positive = pushes toward home win, negative toward away.",
    "brier": "Brier score: mean squared error of probabilities. Lower is better. 0.25 = random.",
    "stage 1": "Stage 1 player embedding model: PyTorch neural network that encodes each batter's EWMA stats + bio into a 16-dim vector, aggregates the 9-man lineup with batting-order weights, and produces 17 game-level features for Stage 2.",
    "xwoba": "Expected Weighted On-Base Average: Statcast metric that assigns a value to each batted ball based on exit velocity and launch angle, removing defense/luck.",
    "barrel": "Barrel rate: % of batted balls with optimal exit velocity (≥98 mph) and launch angle (26–30°). Strongly predicts extra-base hits.",
    "platoon": "Platoon advantage: opposite-hand batter vs. pitcher matchup. Lefty batters typically hit better vs. righty pitchers and vice versa.",
}

# One-paragraph model descriptions (for get_model_info and documentation)
MODEL_DOCS: dict[str, str] = {
    "logistic": "Logistic regression baseline. Linear combination of features passed through sigmoid. Z-score standardised, L2 regularisation. Interpretable coefficients.",
    "lightgbm": "LightGBM gradient boosted trees. Leaf-wise growth, histogram binning. Captures non-linear interactions. Optuna-tuned (num_leaves, learning_rate, n_estimators).",
    "xgboost": "XGBoost gradient boosted trees. Different regularisation from LightGBM. Often best single-model Brier. Optuna-tuned (max_depth, learning_rate, n_estimators).",
    "catboost": "CatBoost gradient boosted trees. Ordered boosting and symmetric trees. Third tree model for ensemble diversity. Optuna-tuned.",
    "mlp": "Multi-layer perceptron: 3 hidden layers (128→64→32), ReLU, Adam. Z-score normalised inputs. L2 weight decay. Captures different non-linear patterns than trees.",
    "stacked": "Production default. Meta-learner (logistic) on top of five calibrated base model probabilities. No raw features. Disjoint calibration/meta split to prevent leakage. C=0.5. v4 adds 17 Stage 1 player features from the neural embedding model.",
}


def get_feature_description(feature_name: str) -> str:
    """Return a short description for a feature, or the name if unknown."""
    key = (feature_name or "").strip()
    return FEATURE_LABELS.get(key, key or "unknown feature")


def get_glossary_term(term: str) -> str | None:
    """Return the glossary definition for a term (lowercased), or None if not found."""
    key = (term or "").strip().lower()
    return GLOSSARY.get(key)


def get_model_docs(model_type: str) -> str:
    """Return a one-paragraph description for a model type, or a generic message."""
    key = (model_type or "").strip().lower()
    return MODEL_DOCS.get(key, f"No specific documentation for model '{model_type}'.")
