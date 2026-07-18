from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


FINAL_STATUSES = frozenset({"FT", "AET", "PEN"})


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _optional_goal(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer or null")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer or null") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer or null")
    return parsed


@dataclass(frozen=True, slots=True)
class Fixture:
    fixture_id: int
    league_id: int
    season: int
    kickoff_utc: int
    home_team_id: int
    away_team_id: int
    home_name: str
    away_name: str
    status: str
    home_goals: int | None
    away_goals: int | None

    @property
    def is_final(self) -> bool:
        return (
            self.status in FINAL_STATUSES
            and self.home_goals is not None
            and self.away_goals is not None
        )

    @property
    def outcome(self) -> int:
        """Return 0=home, 1=draw, 2=away for a verified final fixture."""
        if not self.is_final:
            raise ValueError("outcome is only available for final fixtures")
        assert self.home_goals is not None and self.away_goals is not None
        if self.home_goals > self.away_goals:
            return 0
        if self.home_goals == self.away_goals:
            return 1
        return 2

    @property
    def kickoff_iso(self) -> str:
        return datetime.fromtimestamp(self.kickoff_utc, timezone.utc).isoformat()

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "Fixture":
        fixture = payload.get("fixture") or {}
        league = payload.get("league") or {}
        teams = payload.get("teams") or {}
        goals = payload.get("goals") or {}
        status = fixture.get("status") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        timestamp = fixture.get("timestamp")
        if timestamp is None and fixture.get("date"):
            timestamp = int(
                datetime.fromisoformat(str(fixture["date"]).replace("Z", "+00:00")).timestamp()
            )
        season = league.get("season")
        if isinstance(season, bool):
            raise ValueError("league.season must be an integer")
        try:
            season_value = int(season)
        except (TypeError, ValueError) as exc:
            raise ValueError("league.season must be an integer") from exc

        home_name = str(home.get("name") or "").strip()
        away_name = str(away.get("name") or "").strip()
        if not home_name or not away_name:
            raise ValueError("both team names are required")

        return cls(
            fixture_id=_positive_int(fixture.get("id"), "fixture.id"),
            league_id=_positive_int(league.get("id"), "league.id"),
            season=season_value,
            kickoff_utc=_positive_int(timestamp, "fixture.timestamp"),
            home_team_id=_positive_int(home.get("id"), "teams.home.id"),
            away_team_id=_positive_int(away.get("id"), "teams.away.id"),
            home_name=home_name,
            away_name=away_name,
            status=str(status.get("short") or "").strip().upper(),
            home_goals=_optional_goal(goals.get("home"), "goals.home"),
            away_goals=_optional_goal(goals.get("away"), "goals.away"),
        )


def parse_api_fixtures(payloads: Iterable[dict[str, Any]]) -> tuple[list[Fixture], list[str]]:
    fixtures: list[Fixture] = []
    errors: list[str] = []
    for index, payload in enumerate(payloads):
        try:
            fixtures.append(Fixture.from_api(payload))
        except (TypeError, ValueError) as exc:
            errors.append(f"fixture[{index}]: {exc}")
    return fixtures, errors

