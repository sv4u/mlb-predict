# MLB Win Probability Model

A research-grade, reproducible MLB win probability modeling platform covering seasons 2000–2025. The system is currently in its ingestion and crosswalk phase, with modeling, scoring, and drift monitoring planned as future layers.

## Features

- MLB Stats API wrapper (async, token-bucket rate limiting, SHA256-keyed disk cache)
- Regular-season schedule ingestion (chunked by month with adaptive binary splitting)
- Retrosheet GL game log ingestion with automatic source fallback:
  - Chadwick Bureau GitHub mirror (primary)
  - Retrosheet.org ZIP archives (fallback)
- Retrosheet ↔ MLB `game_pk` crosswalk builder with coverage reports (≥ 99% threshold enforced)
- Deterministic provenance: every artifact is SHA256-checksummed

## Install

Requires Python ≥ 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

To also install development and test dependencies:

```bash
pip install -e ".[dev]"
```

## Required mapping file

Before running the crosswalk, create the team ID mapping file:

```bash
cp docs/team_id_map_template.csv data/processed/team_id_map_retro_to_mlb.csv
# Edit the file to add all team codes for your target seasons
```

## Run (single season)

```bash
python scripts/ingest_schedule.py --seasons 2025
python scripts/ingest_retrosheet_gamelogs.py --seasons 2025
python scripts/build_crosswalk.py --seasons 2025
```

## Full ingest (2000–2025 + current year)

```bash
python scripts/ingest_all.py
```

## Testing

The test suite uses `pytest`, `pytest-asyncio`, `hypothesis`, and `aioresponses`.

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

The suite contains 132 tests organized into:

| Module | Description |
|---|---|
| `tests/unit/test_hashing.py` | SHA256 file and aggregate utilities |
| `tests/unit/test_token_bucket.py` | Rate limiter correctness and concurrency |
| `tests/unit/test_mlbapi_client.py` | HTTP client: caching, retries, 404/429/5xx |
| `tests/unit/test_schedule.py` | Schedule normalization and async fetch functions |
| `tests/unit/test_teams.py` | Team map building and async fetch functions |
| `tests/unit/test_gamelogs.py` | Retrosheet download, ZIP extraction, parsing |
| `tests/unit/test_id_map.py` | Retrosheet team ID crosswalk loader |
| `tests/unit/test_crosswalk.py` | Game-level crosswalk builder and coverage metrics |
| `tests/property/test_properties.py` | Hypothesis property-based invariants |

## Project structure

```
src/winprob/
├── mlbapi/       # Async MLB Stats API client (rate-limited, cached)
├── retrosheet/   # Retrosheet game log download and parsing
├── ingest/       # Retrosheet-to-MLB team ID map loader
├── crosswalk/    # Game-level crosswalk builder
└── util/         # SHA256 hashing utilities

scripts/
├── ingest_schedule.py            # Schedule ingestion pipeline
├── ingest_retrosheet_gamelogs.py # Retrosheet ingestion pipeline
├── build_crosswalk.py            # Crosswalk builder
└── ingest_all.py                 # Orchestrates all three in parallel

tests/
├── unit/         # Example-based unit tests (one per module)
└── property/     # Hypothesis property-based tests
```

## Attribution

Retrosheet data is subject to Retrosheet's notice. See `docs/RETROSHEET_ATTRIBUTION.md`.
