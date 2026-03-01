# DATA_SCHEMA.md
## MLB Win Probability Modeling System — Data Contracts

This document defines on-disk schemas and invariants for all persisted datasets.

---

# 1. Conventions

## 1.1 Storage formats

Unless otherwise stated, each dataset is written in **both**:

- **Parquet** (authoritative, typed)
- **CSV** (human inspection)

Parquet files MUST be produced with stable column ordering and deterministic row ordering when feasible.

## 1.2 Paths

All paths are relative to repository root.

- Raw: `data/raw/**`
- Processed: `data/processed/**`

## 1.3 Identifiers

- `game_pk` (MLB Stats API) is the canonical game identifier wherever available.
- Crosswalk tables map Retrosheet game keys → MLB `game_pk`.

## 1.4 Hashing / provenance

Where a `*.checksum.json` file exists, it MUST include:

- Row counts
- SHA256 of output artifacts
- Source selection fields (when applicable)
- Any configuration that affects outputs

---

# 2. Raw Data Layout

## 2.1 MLB Stats API cache

Base: `data/raw/mlb_api/`

Layout:

- `data/raw/mlb_api/<endpoint>/<cache_key>.json`
- `data/raw/mlb_api/metadata.jsonl` (append-only)

### 2.1.1 `metadata.jsonl` schema

Each line is a JSON object:

| Field | Type | Required | Notes |
|---|---:|---:|---|
| `ts_unix` | float | yes | Unix timestamp |
| `url` | string | yes | Request URL |
| `params` | object | yes | Query params |
| `cache_key` | string | yes | SHA256 over endpoint+params |
| `endpoint` | string | yes | Endpoint name |
| `status` | int | yes | HTTP status |

Invariants:
- `metadata.jsonl` is append-only.
- cached payload filenames are deterministic from `(endpoint, params)`.

## 2.2 Retrosheet raw game logs

Base: `data/raw/retrosheet/gamelogs/`

- `GL<season>.TXT` (raw text)

---

# 3. Processed Data Schemas

## 3.1 Teams

Path:
- `data/processed/teams/teams_<season>.parquet`

Schema:

| Column | Type | Required | Notes |
|---|---|---:|---|
| `season` | int32 | yes | season requested |
| `mlb_team_id` | int32 | yes | Stats API team id |
| `abbrev` | string | yes | e.g., LAD |
| `name` | string | yes | team name |

Invariants:
- `mlb_team_id` unique per season.

## 3.2 Schedule (regular season only)

Paths:
- `data/processed/schedule/games_<season>.parquet`
- `data/processed/schedule/games_<season>.csv`
- `data/processed/schedule/games_<season>.checksum.json`

Schema:

| Column | Type | Required | Notes |
|---|---|---:|---|
| `game_pk` | int64 | yes | canonical game id |
| `season` | int32 | yes | season |
| `game_date_utc` | string | yes | ISO 8601 |
| `game_date_local` | string | no | ISO 8601, derived via venue tz |
| `home_mlb_id` | int32 | yes | team id |
| `away_mlb_id` | int32 | yes | team id |
| `home_abbrev` | string | yes | from teams endpoint |
| `away_abbrev` | string | yes | from teams endpoint |
| `venue_id` | int32 | no | venue id |
| `local_timezone` | string | no | IANA tz |
| `double_header` | string | no | Stats API field |
| `game_number` | int32? | no | DH number when present |
| `status` | string | no | e.g., Scheduled |

Invariants:
- Unique `game_pk`.
- Dataset includes only `gameType=R`.

Checksum schema (minimum):

| Field | Type | Required |
|---|---|---:|
| `season` | int | yes |
| `row_count` | int | yes |
| `parquet_sha256` | string | yes |
| `csv_sha256` | string | yes |
| `raw_payloads_sha256` | string\|null | yes |
| `raw_file_count` | int | yes |
| `max_response_mb` | int | yes |
| `max_split_depth` | int | yes |
| `mlbapi_config` | object | yes |

## 3.3 Retrosheet game logs (GL)

Paths:
- `data/processed/retrosheet/gamelogs_<season>.parquet`
- `data/processed/retrosheet/gamelogs_<season>.csv`
- `data/processed/retrosheet/gamelogs_<season>.checksum.json`

Schema:
- Columns follow the Retrosheet GL format (see code for full list).
- The following normalized columns MUST exist and be parseable:

| Column | Type | Required |
|---|---|---:|
| `date` | date | yes |
| `game_num` | int32? | yes |
| `visiting_team` | string | yes |
| `home_team` | string | yes |
| `visiting_score` | int32? | no |
| `home_score` | int32? | no |
| `visiting_starting_pitcher_id` | string | no |
| `home_starting_pitcher_id` | string | no |

Checksum schema (minimum):

| Field | Type | Required |
|---|---|---:|
| `season` | int | yes |
| `row_count` | int | yes |
| `raw_sha256` | string | yes |
| `parquet_sha256` | string | yes |
| `csv_sha256` | string | yes |
| `raw_path` | string | yes |
| `source_used` | string | yes |
| `url_used` | string\|null | yes |
| `fallback_reason` | string\|null | yes |

## 3.4 Crosswalk (Retrosheet → MLB)

Paths:
- `data/processed/crosswalk/game_id_map_<season>.parquet`
- `data/processed/crosswalk/unresolved_<season>.parquet`
- `data/processed/crosswalk/crosswalk_coverage_report.parquet`
- `data/processed/crosswalk/crosswalk_coverage_report.csv`

Schema (`game_id_map_<season>.parquet`):

| Column | Type | Required |
|---|---|---:|
| `date` | date | yes |
| `home_mlb_id` | int32 | yes |
| `away_mlb_id` | int32 | yes |
| `home_retro` | string | yes |
| `away_retro` | string | yes |
| `dh_game_num` | int32? | no |
| `status` | string | yes | matched/missing/ambiguous |
| `mlb_game_pk` | int64? | no |
| `match_confidence` | float | yes |
| `notes` | string | yes |

Coverage invariants:
- Minimum required: **99.0% matched**.
- Any season below threshold MUST be flagged in `crosswalk_seasons_below_threshold.csv`.

## 3.5 Prediction snapshots (planned contract)

Path template:
- `data/processed/predictions/season=YYYY/snapshots/run_ts=<iso>.parquet`

Schema (minimum):

| Column | Type | Required |
|---|---|---:|
| `game_pk` | int64 | yes |
| `season` | int32 | yes |
| `predicted_home_win_prob` | float | yes |
| `run_ts_utc` | timestamp | yes |
| `model_version` | string | yes |
| `schedule_hash` | string | yes |
| `feature_hash` | string | yes |
| `lineup_param_hash` | string | yes |
| `starter_param_hash` | string | yes |
| `git_commit` | string | yes |
| `tag` | string\|null | no |

Immutability:
- Snapshot files MUST never be overwritten.

## 3.6 Drift artifacts (planned contract)

Per-season metrics:
- `data/processed/predictions/season=YYYY/run_metrics.parquet`

Global metrics:
- `data/processed/predictions/global_run_metrics.parquet`

Diff artifacts (recommended):
- incremental and baseline diffs stored as Parquet with stable schema:
  - `game_pk`, `p_old`, `p_new`, `delta`, `abs_delta`, `direction`

---

# 4. Manual Mapping File Contract

Path:
- `data/processed/team_id_map_retro_to_mlb.csv`

Schema:

| Column | Type | Required | Notes |
|---|---|---:|---|
| `retro_team_code` | string | yes | e.g., LAN |
| `mlb_team_id` | int32 | yes | Stats API team id |
| `mlb_abbrev` | string | no | convenience |
| `valid_from_season` | int32 | yes | inclusive |
| `valid_to_season` | int32 | yes | inclusive |

Invariants:
- For any `(retro_team_code, season)` there MUST be exactly one mapping row.
- Gaps or overlaps are errors.

---

# 5. Determinism Requirements

Any module generating derived data MUST:
- sort rows deterministically before writing
- avoid unordered set/dict iteration in output generation
- record config and hashes sufficient to reproduce
