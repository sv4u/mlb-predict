"""Live lineup fetcher from the MLB Stats API.

Fetches confirmed batting orders for current-season games via the
``/game/{game_pk}/boxscore`` endpoint.  Returns lineup data with player
MLBAM IDs, batting order positions, and batting side.

For historical training, Retrosheet gamelogs provide lineup data via
``home_1_id..home_9_id`` columns.  This module is used only for
current-season scoring when Retrosheet data is not yet available.

All requests are routed through the existing async MLBAPIClient
(rate-limited, cached).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mlb_predict.mlbapi.client import MLBAPIClient

logger = logging.getLogger(__name__)


@dataclass
class LineupEntry:
    """A single batter in a confirmed lineup."""

    mlbam_id: int
    full_name: str
    batting_order: int
    position: str
    bat_side: str


@dataclass
class GameLineup:
    """Confirmed lineups for both teams in a game."""

    game_pk: int
    home_lineup: list[LineupEntry]
    away_lineup: list[LineupEntry]
    home_sp_id: int | None
    away_sp_id: int | None
    home_sp_name: str
    away_sp_name: str


async def fetch_game_lineup(client: MLBAPIClient, game_pk: int) -> GameLineup | None:
    """Fetch confirmed lineups for a game from the MLB Stats API boxscore.

    Returns None if the lineup is not yet posted or the endpoint fails.
    """
    try:
        data = await client.get_json(
            f"/game/{game_pk}/boxscore",
            params={},
        )
    except Exception as exc:
        logger.warning("Failed to fetch lineup for game_pk=%d: %s", game_pk, exc)
        return None

    teams = data.get("teams", {})
    home_data = teams.get("home", {})
    away_data = teams.get("away", {})

    home_lineup = _extract_lineup(home_data)
    away_lineup = _extract_lineup(away_data)

    if not home_lineup or not away_lineup:
        return None

    home_sp_id, home_sp_name = _extract_sp(home_data)
    away_sp_id, away_sp_name = _extract_sp(away_data)

    return GameLineup(
        game_pk=game_pk,
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        home_sp_id=home_sp_id,
        away_sp_id=away_sp_id,
        home_sp_name=home_sp_name,
        away_sp_name=away_sp_name,
    )


def _extract_lineup(team_data: dict[str, Any]) -> list[LineupEntry]:
    """Extract batting order from the boxscore team data."""
    batters = team_data.get("battingOrder", [])
    players = team_data.get("players", {})

    entries: list[LineupEntry] = []
    for i, batter_id in enumerate(batters):
        player_key = f"ID{batter_id}"
        player_info = players.get(player_key, {})
        person = player_info.get("person", {})
        position = player_info.get("position", {})
        bat_side_data = player_info.get("batSide", person.get("batSide", {}))

        entries.append(
            LineupEntry(
                mlbam_id=int(batter_id),
                full_name=person.get("fullName", ""),
                batting_order=i + 1,
                position=position.get("abbreviation", ""),
                bat_side=bat_side_data.get("code", "R") if isinstance(bat_side_data, dict) else "R",
            )
        )

    return entries


def _extract_sp(team_data: dict[str, Any]) -> tuple[int | None, str]:
    """Extract starting pitcher ID and name from boxscore data."""
    pitchers = team_data.get("pitchers", [])
    players = team_data.get("players", {})

    if not pitchers:
        return None, ""

    sp_id = pitchers[0]
    player_key = f"ID{sp_id}"
    player_info = players.get(player_key, {})
    person = player_info.get("person", {})
    return int(sp_id), person.get("fullName", "")


def lineup_to_player_ids(lineup: list[LineupEntry]) -> list[int]:
    """Extract MLBAM IDs from a lineup in batting order."""
    return [entry.mlbam_id for entry in sorted(lineup, key=lambda e: e.batting_order)]
