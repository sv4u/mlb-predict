from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .client import MLBAPIClient


@dataclass(frozen=True)
class TeamMaps:
    mlb_id_to_abbrev: dict[int, str]
    abbrev_to_mlb_id: dict[str, int]
    mlb_id_to_name: dict[int, str]


_TEAMS_COLS = ["season", "mlb_team_id", "abbrev", "name"]


async def get_teams_df(client: MLBAPIClient, *, season: int, sport_id: int = 1) -> pd.DataFrame:
    params: Mapping[str, Any] = {"sportId": sport_id, "season": season}
    raw = await client.get_json("teams", params)
    rows: list[dict[str, Any]] = []
    for t in raw.get("teams", []):
        rows.append(
            {
                "season": season,
                "mlb_team_id": t.get("id"),
                "abbrev": t.get("abbreviation"),
                "name": t.get("name"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=_TEAMS_COLS)
    df = pd.DataFrame(rows).dropna(subset=["mlb_team_id"])
    if df.empty:
        return pd.DataFrame(columns=_TEAMS_COLS)
    df["mlb_team_id"] = df["mlb_team_id"].astype(int)
    return df


def build_team_maps(df: pd.DataFrame) -> TeamMaps:
    id_to_abbrev = {int(r.mlb_team_id): str(r.abbrev) for r in df.itertuples(index=False)}
    abbrev_to_id = {str(r.abbrev): int(r.mlb_team_id) for r in df.itertuples(index=False)}
    id_to_name = {int(r.mlb_team_id): str(r.name) for r in df.itertuples(index=False)}
    return TeamMaps(id_to_abbrev, abbrev_to_id, id_to_name)
