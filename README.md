# MLB Win Probability Model (Source Bundle)

This bundle provides a reproducible ingestion + crosswalk foundation:

- MLB Stats API wrapper (async, token-bucket, cached)
- Regular-season schedule ingestion (chunked by month, adaptive split)
- Retrosheet GL game log ingestion with automatic fallback:
  - Chadwick Bureau GitHub mirror (primary)
  - Retrosheet.org ZIP archives (fallback)
- Retrosheet ↔ MLB `game_pk` crosswalk builder (coverage reports)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Required mapping file

Create `data/processed/team_id_map_retro_to_mlb.csv` using `docs/team_id_map_template.csv`.

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

## Attribution

Retrosheet data is subject to Retrosheet’s notice. See `docs/RETROSHEET_ATTRIBUTION.md`.
