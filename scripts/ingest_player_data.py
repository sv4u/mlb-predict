"""Ingest player-level data: FanGraphs stats, Statcast stats, and biographical data.

Fetches per-player season stats from FanGraphs (2002+) and Statcast (2015+),
plus biographical data from the Chadwick register.  All data is cached as
Parquet files under ``data/processed/player/``.

Usage
-----
    python scripts/ingest_player_data.py                    # all available seasons
    python scripts/ingest_player_data.py --start 2015       # from 2015 onwards
    python scripts/ingest_player_data.py --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    """Ingest player data for the specified seasons."""
    ap = argparse.ArgumentParser(description="Ingest player-level stats and biographical data")
    ap.add_argument(
        "--start",
        type=int,
        default=2002,
        help="First season to ingest (default: 2002)",
    )
    ap.add_argument(
        "--end",
        type=int,
        default=datetime.now(timezone.utc).year,
        help="Last season to ingest (default: current year)",
    )
    ap.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        help="Explicit list of seasons (overrides --start/--end)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/player"),
        help="Output directory for cached player data",
    )
    ap.add_argument(
        "--skip-bio",
        action="store_true",
        help="Skip biographical data ingestion",
    )
    ap.add_argument(
        "--skip-batters",
        action="store_true",
        help="Skip batter stat ingestion",
    )
    ap.add_argument(
        "--skip-pitchers",
        action="store_true",
        help="Skip pitcher stat ingestion",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seasons = args.seasons or list(range(args.start, args.end + 1))
    print(f"Ingesting player data for {len(seasons)} seasons: {seasons[0]}–{seasons[-1]}")
    print(f"Output directory: {out}")

    # --- Biographical data ---------------------------------------------------
    if not args.skip_bio:
        print("\n[1/3] Building biographical data…")
        from mlb_predict.player.biographical import build_biographical_df

        bio_df = build_biographical_df(cache_dir=out)
        print(f"  Biographical data: {len(bio_df)} players")

    # --- Batter stats --------------------------------------------------------
    if not args.skip_batters:
        print("\n[2/3] Ingesting batter stats…")
        from mlb_predict.player.ingestion import get_batter_stats_for_season

        for s in seasons:
            df = get_batter_stats_for_season(s, cache_dir=out)
            n = len(df) if not df.empty else 0
            print(f"  {s}: {n} batters")

    # --- Pitcher stats -------------------------------------------------------
    if not args.skip_pitchers:
        print("\n[3/3] Ingesting pitcher stats…")
        from mlb_predict.player.ingestion import get_pitcher_stats_for_season

        for s in seasons:
            df = get_pitcher_stats_for_season(s, cache_dir=out)
            n = len(df) if not df.empty else 0
            print(f"  {s}: {n} pitchers")

    print("\nDone.")


if __name__ == "__main__":
    main()
