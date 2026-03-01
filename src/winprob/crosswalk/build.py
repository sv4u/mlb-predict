from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from winprob.ingest.id_map import RetroTeamMap


@dataclass(frozen=True)
class CrosswalkResult:
    df: pd.DataFrame
    coverage_pct: float
    matched: int
    missing: int
    ambiguous: int


def _prep_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    s = schedule.copy()
    s["date"] = pd.to_datetime(s["game_date_utc"].str.replace("Z", "+00:00", regex=False), errors="coerce").dt.date
    s["game_number"] = pd.to_numeric(s["game_number"], errors="coerce").astype("Int64")
    return s


_CROSSWALK_COLS = [
    "date", "home_mlb_id", "away_mlb_id", "home_retro", "away_retro",
    "dh_game_num", "status", "mlb_game_pk", "match_confidence", "notes",
]


def build_crosswalk(*, season: int, schedule: pd.DataFrame, gamelogs: pd.DataFrame, retro_team_map: RetroTeamMap) -> CrosswalkResult:
    sched = _prep_schedule(schedule)

    gl = gamelogs.copy().dropna(subset=["date", "home_team", "visiting_team"])
    if gl.empty:
        return CrosswalkResult(
            df=pd.DataFrame(columns=_CROSSWALK_COLS),
            coverage_pct=0.0, matched=0, missing=0, ambiguous=0,
        )
    gl["home_mlb_id"] = gl["home_team"].map(lambda x: retro_team_map.retro_to_mlb_id(str(x), season))
    gl["away_mlb_id"] = gl["visiting_team"].map(lambda x: retro_team_map.retro_to_mlb_id(str(x), season))
    gl["dh_game_num"] = pd.to_numeric(gl["game_num"], errors="coerce").astype("Int64")

    merged = gl.merge(
        sched[["game_pk", "date", "home_mlb_id", "away_mlb_id", "game_number", "venue_id"]],
        on=["date", "home_mlb_id", "away_mlb_id"],
        how="left",
    )

    def resolve_group(g: pd.DataFrame) -> pd.DataFrame:
        if g["game_pk"].isna().all():
            return g.head(1).assign(status="missing", mlb_game_pk=pd.NA, match_confidence=0.0, notes="no_schedule_match")
        cands = g.dropna(subset=["game_pk"])
        if cands["game_pk"].nunique() == 1:
            pk = int(cands["game_pk"].iloc[0])
            return g.head(1).assign(status="matched", mlb_game_pk=pk, match_confidence=1.0, notes="unique")
        if pd.notna(g["dh_game_num"].iloc[0]):
            m = cands[cands["game_number"] == g["dh_game_num"].iloc[0]]
            if m["game_pk"].nunique() == 1:
                pk = int(m["game_pk"].iloc[0])
                return g.head(1).assign(status="matched", mlb_game_pk=pk, match_confidence=0.9, notes="matched_on_game_number")
        return g.head(1).assign(status="ambiguous", mlb_game_pk=pd.NA, match_confidence=0.0, notes="multiple_candidates")

    key_cols = ["date", "home_mlb_id", "away_mlb_id", "home_team", "visiting_team", "dh_game_num"]
    resolved = merged.groupby(key_cols, dropna=False, as_index=False).apply(resolve_group).reset_index(drop=True)

    out = resolved[[
        "date",
        "home_mlb_id",
        "away_mlb_id",
        "home_team",
        "visiting_team",
        "dh_game_num",
        "status",
        "mlb_game_pk",
        "match_confidence",
        "notes",
    ]].rename(columns={"home_team": "home_retro", "visiting_team": "away_retro"})

    matched = int((out["status"] == "matched").sum())
    ambiguous = int((out["status"] == "ambiguous").sum())
    missing = int((out["status"] == "missing").sum())
    coverage_pct = 100.0 * matched / max(len(out), 1)

    return CrosswalkResult(df=out, coverage_pct=coverage_pct, matched=matched, missing=missing, ambiguous=ambiguous)
