# MODEL_SPEC.md
## MLB Win Probability Modeling System — Modeling Specification

This document defines the mathematical and engineering contracts for modeling win probabilities.

---

# 1. Problem Definition

For each scheduled MLB regular-season game `g`, estimate:

- `P(HomeWin | x_g)`

where `x_g` includes team- and player-level features available at prediction time.

Outputs are probabilities in `[0, 1]` suitable for evaluation (log loss / Brier), calibration, and drift tracking.

---

# 2. Baseline Model (MVP)

## 2.1 Logistic Regression

Let `x_g ∈ ℝ^d` be a feature vector.

- `p_g = σ(wᵀ x_g + b)`
- `σ(z) = 1 / (1 + exp(-z))`

Train by minimizing regularized negative log likelihood:

- `L = - Σ_g [ y_g log(p_g) + (1-y_g) log(1-p_g) ] + λ ||w||²`

Where:
- `y_g = 1` if home wins, else `0`
- `λ` is the L2 regularization weight

## 2.2 Output Contract

The model MUST output:

- `predicted_home_win_prob` for each `game_pk`
- metadata for reproducibility:
  - `model_version`
  - training seasons/cutoff
  - feature_set_version
  - hyperparameters

---

# 3. Feature Engineering Contract

Features must satisfy:

- derived only from data available at scoring time (no leakage)
- deterministic computation
- stable schema + versioning

## 3.1 Feature categories (planned)

MVP feature sets (priority order):

1. Team strength
   - rolling run differential
   - rolling offense proxy
   - rolling pitching proxy

2. Starting pitcher
   - rolling starter performance
   - handedness splits (optional)

3. Rest/travel
   - days rest
   - home/away indicator
   - travel/time zone (optional)

4. Lineup-based (long-term)
   - expected lineup aggregates
   - platoon matchup aggregates

---

# 4. Training & Evaluation Protocol

## 4.1 Splits

Default:
- Train: seasons 2000–(N-1)
- Test: season N

Recommended alternative:
- rolling-window validation (last K seasons)

## 4.2 Metrics

Mandatory:
- Log loss
- Brier score

Recommended:
- calibration curve + ECE
- AUC (secondary)

## 4.3 Calibration (planned)

Options:
- Platt scaling
- Isotonic regression

Calibration must be versioned separately and trained on held-out data.

---

# 5. Lineup-Aware Modeling (Planned)

Lineup-aware mode incorporates expected lineups:

Requirements:
- expected lineups timestamped
- scoring reproducible given lineup snapshot
- store `lineup_param_hash` and lineup snapshot identifiers

---

# 6. Pitcher-Aware Modeling (Planned)

Pitcher module requirements:
- projected starters with pitcher ids
- deterministic starter selection rules
- store `starter_param_hash`

---

# 7. Simulation (Future / Optional)

If Monte Carlo is introduced:
- RNG must be seeded
- seed must be stored in metadata
- simulation outputs must be reproducible

---

# 8. Versioning and Reproducibility

## 8.1 Semantic versioning

Recommended:
- major: feature schema changes
- minor: new features or calibration approach
- patch: bug fixes without schema changes

## 8.2 Required hashes per snapshot

- `schedule_hash`
- `feature_hash`
- `lineup_param_hash`
- `starter_param_hash`

Hashes must be computed from canonical, sorted inputs.

---

# 9. Integrity Constraints

Agents MUST NOT:
- train on test-season outcomes when generating predictions for that season
- leak future lineup/starter information into earlier snapshots
- overwrite model artifacts without version bump
- alter historical prediction snapshots
