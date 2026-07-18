from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import groupby
from typing import Deque, Iterable

from .domain import Fixture


FEATURE_SCHEMA_VERSION = "proofxi-asof-v2"
DEFAULT_PUBLICATION_LEAD_SECONDS = 86_400
DEFAULT_RESULT_AVAILABILITY_LAG_SECONDS = 86_400
FEATURE_NAMES = (
    "home_points_sum_5",
    "away_points_sum_5",
    "home_goals_for_sum_5",
    "home_goals_against_sum_5",
    "away_goals_for_sum_5",
    "away_goals_against_sum_5",
    "home_home_points_sum_5",
    "home_home_goal_diff_sum_5",
    "away_away_points_sum_5",
    "away_away_goal_diff_sum_5",
    "elo_diff_home_adv",
    "h2h_home_points_sum_5",
    "home_rest_days",
    "away_rest_days",
)


@dataclass(frozen=True, slots=True)
class MatchView:
    kickoff_utc: int
    available_utc: int
    opponent_id: int
    points: int
    goals_for: int
    goals_against: int


@dataclass(frozen=True, slots=True)
class H2HView:
    kickoff_utc: int
    available_utc: int
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    fixture: Fixture
    as_of_utc: int
    source_max_match_utc: int
    values: dict[str, float | None]

    def json_values(self) -> dict[str, float | None]:
        return {name: self.values[name] for name in FEATURE_NAMES}


@dataclass(frozen=True, slots=True)
class TrainingExample:
    fixture: Fixture
    snapshot: FeatureSnapshot
    target: int
    target_available_utc: int


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _sum_or_none(items: list[MatchView], field: str) -> float | None:
    if not items:
        return None
    return float(sum(getattr(item, field) for item in items))


class FeatureEngine:
    """Build chronological features without consulting the target result.

    Historical snapshots use a conservative publication lead and result-
    availability lag. Fixtures sharing a kickoff timestamp are snapshotted as
    one batch. This engine does not reconstruct correction observed-at history;
    that requires a separately versioned source-revision feed.
    """

    def __init__(
        self,
        *,
        window: int = 5,
        elo_base: float = 1500.0,
        elo_k: float = 20.0,
        home_advantage: float = 65.0,
        season_carry: float = 0.75,
        max_rest_days: float = 30.0,
        publication_lead_seconds: int = DEFAULT_PUBLICATION_LEAD_SECONDS,
        result_availability_lag_seconds: int = DEFAULT_RESULT_AVAILABILITY_LAG_SECONDS,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0 <= season_carry <= 1:
            raise ValueError("season_carry must be between 0 and 1")
        if publication_lead_seconds < 0:
            raise ValueError("publication_lead_seconds cannot be negative")
        if result_availability_lag_seconds < 0:
            raise ValueError("result_availability_lag_seconds cannot be negative")
        self.window = window
        self.elo_base = elo_base
        self.elo_k = elo_k
        self.home_advantage = home_advantage
        self.season_carry = season_carry
        self.max_rest_days = max_rest_days
        self.publication_lead_seconds = publication_lead_seconds
        self.result_availability_lag_seconds = result_availability_lag_seconds
        self._all: dict[tuple[int, int], Deque[MatchView]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._home: dict[tuple[int, int], Deque[MatchView]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._away: dict[tuple[int, int], Deque[MatchView]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._h2h: dict[tuple[int, int, int], Deque[H2HView]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._elo: dict[tuple[int, int], float] = {}
        self._last_match: dict[tuple[int, int], int] = {}
        self._league_season: dict[int, int] = {}

    def build(
        self,
        fixtures: Iterable[Fixture],
        *,
        prediction_as_of_utc: int | None = None,
    ) -> tuple[list[TrainingExample], list[FeatureSnapshot]]:
        ordered = sorted(fixtures, key=lambda item: (item.kickoff_utc, item.fixture_id))
        training: list[TrainingExample] = []
        snapshots: list[FeatureSnapshot] = []
        pending: list[tuple[int, int, int, int, Fixture]] = []
        pending_sequence = 0

        for kickoff_utc, grouped in groupby(ordered, key=lambda item: item.kickoff_utc):
            batch = list(grouped)
            as_of_utc = (
                prediction_as_of_utc
                if prediction_as_of_utc is not None and kickoff_utc > prediction_as_of_utc
                else kickoff_utc - self.publication_lead_seconds
            )

            # A historical final becomes usable only after its declared
            # availability proxy, never at kick-off. Strict inequality matches
            # the stored feature-snapshot contract: a source timestamp equal to
            # the prediction timestamp is not eligible.
            while pending and pending[0][0] < as_of_utc:
                available_utc, _, _, _, available_fixture = heapq.heappop(pending)
                self._apply_final(available_fixture, available_utc=available_utc)

            for league_id in sorted({item.league_id for item in batch}):
                season = min(item.season for item in batch if item.league_id == league_id)
                self._start_season(league_id, season)

            batch_snapshots: list[FeatureSnapshot] = []
            for fixture in batch:
                snapshot = self._snapshot(fixture, as_of_utc)
                batch_snapshots.append(snapshot)
                snapshots.append(snapshot)
                if fixture.is_final:
                    training.append(
                        TrainingExample(
                            fixture=fixture,
                            snapshot=snapshot,
                            target=fixture.outcome,
                            target_available_utc=(
                                fixture.kickoff_utc
                                + self.result_availability_lag_seconds
                            ),
                        )
                    )

            # Same-time results are queued only after every snapshot in the
            # batch is produced. Non-final rows never mutate feature state.
            for fixture in batch:
                if fixture.is_final:
                    available_utc = (
                        fixture.kickoff_utc + self.result_availability_lag_seconds
                    )
                    heapq.heappush(
                        pending,
                        (
                            available_utc,
                            fixture.kickoff_utc,
                            fixture.fixture_id,
                            pending_sequence,
                            fixture,
                        ),
                    )
                    pending_sequence += 1

        return training, snapshots

    def _start_season(self, league_id: int, season: int) -> None:
        previous = self._league_season.get(league_id)
        if previous is None:
            self._league_season[league_id] = season
            return
        if previous == season:
            return
        if season < previous:
            raise ValueError(
                f"fixtures are not season-chronological for league {league_id}: {season} < {previous}"
            )
        for (rating_league, team_id), rating in list(self._elo.items()):
            if rating_league == league_id:
                self._elo[(rating_league, team_id)] = self.elo_base + self.season_carry * (
                    rating - self.elo_base
                )
        self._league_season[league_id] = season

    def _rating(self, league_id: int, team_id: int) -> float:
        return self._elo.get((league_id, team_id), self.elo_base)

    def _snapshot(self, fixture: Fixture, as_of_utc: int) -> FeatureSnapshot:
        league = fixture.league_id
        home_key = (league, fixture.home_team_id)
        away_key = (league, fixture.away_team_id)
        home_all = list(self._all[home_key])
        away_all = list(self._all[away_key])
        home_venue = list(self._home[home_key])
        away_venue = list(self._away[away_key])
        pair = tuple(sorted((fixture.home_team_id, fixture.away_team_id)))
        h2h = list(self._h2h[(league, pair[0], pair[1])])
        home_elo = self._rating(league, fixture.home_team_id)
        away_elo = self._rating(league, fixture.away_team_id)

        h2h_points = 0
        for item in h2h:
            if item.home_team_id == fixture.home_team_id:
                goals_for, goals_against = item.home_goals, item.away_goals
            else:
                goals_for, goals_against = item.away_goals, item.home_goals
            h2h_points += _points(goals_for, goals_against)

        source_times = [
            *(item.available_utc for item in home_all),
            *(item.available_utc for item in away_all),
            *(item.available_utc for item in home_venue),
            *(item.available_utc for item in away_venue),
            *(item.available_utc for item in h2h),
        ]
        # The persisted schema currently requires an integer. When no historical
        # source contributed, as_of_utc - 1 is a sentinel rather than an observed
        # source timestamp; callers must use the null feature values to distinguish
        # that state.
        source_max = max(source_times, default=as_of_utc - 1)
        if source_max >= as_of_utc:
            raise ValueError(
                f"as-of leakage for fixture {fixture.fixture_id}: source {source_max} >= {as_of_utc}"
            )

        values: dict[str, float | None] = {
            "home_points_sum_5": _sum_or_none(home_all, "points"),
            "away_points_sum_5": _sum_or_none(away_all, "points"),
            "home_goals_for_sum_5": _sum_or_none(home_all, "goals_for"),
            "home_goals_against_sum_5": _sum_or_none(home_all, "goals_against"),
            "away_goals_for_sum_5": _sum_or_none(away_all, "goals_for"),
            "away_goals_against_sum_5": _sum_or_none(away_all, "goals_against"),
            "home_home_points_sum_5": _sum_or_none(home_venue, "points"),
            "home_home_goal_diff_sum_5": (
                float(sum(item.goals_for - item.goals_against for item in home_venue))
                if home_venue
                else None
            ),
            "away_away_points_sum_5": _sum_or_none(away_venue, "points"),
            "away_away_goal_diff_sum_5": (
                float(sum(item.goals_for - item.goals_against for item in away_venue))
                if away_venue
                else None
            ),
            "elo_diff_home_adv": round(home_elo + self.home_advantage - away_elo, 8),
            "h2h_home_points_sum_5": float(h2h_points) if h2h else None,
            "home_rest_days": self._rest_days(home_key, fixture.kickoff_utc),
            "away_rest_days": self._rest_days(away_key, fixture.kickoff_utc),
        }
        if tuple(values) != FEATURE_NAMES:
            raise AssertionError("feature schema order changed unexpectedly")
        return FeatureSnapshot(
            fixture=fixture,
            as_of_utc=as_of_utc,
            source_max_match_utc=source_max,
            values=values,
        )

    def _rest_days(self, key: tuple[int, int], kickoff_utc: int) -> float | None:
        previous = self._last_match.get(key)
        if previous is None:
            return None
        days = (kickoff_utc - previous) / 86_400
        if days < 0:
            raise ValueError("negative rest interval indicates non-chronological input")
        return round(min(days, self.max_rest_days), 6)

    def _apply_final(self, fixture: Fixture, *, available_utc: int) -> None:
        assert fixture.home_goals is not None and fixture.away_goals is not None
        league = fixture.league_id
        home_key = (league, fixture.home_team_id)
        away_key = (league, fixture.away_team_id)
        home_view = MatchView(
            kickoff_utc=fixture.kickoff_utc,
            available_utc=available_utc,
            opponent_id=fixture.away_team_id,
            points=_points(fixture.home_goals, fixture.away_goals),
            goals_for=fixture.home_goals,
            goals_against=fixture.away_goals,
        )
        away_view = MatchView(
            kickoff_utc=fixture.kickoff_utc,
            available_utc=available_utc,
            opponent_id=fixture.home_team_id,
            points=_points(fixture.away_goals, fixture.home_goals),
            goals_for=fixture.away_goals,
            goals_against=fixture.home_goals,
        )
        self._all[home_key].append(home_view)
        self._all[away_key].append(away_view)
        self._home[home_key].append(home_view)
        self._away[away_key].append(away_view)
        pair = tuple(sorted((fixture.home_team_id, fixture.away_team_id)))
        self._h2h[(league, pair[0], pair[1])].append(
            H2HView(
                kickoff_utc=fixture.kickoff_utc,
                available_utc=available_utc,
                home_team_id=fixture.home_team_id,
                away_team_id=fixture.away_team_id,
                home_goals=fixture.home_goals,
                away_goals=fixture.away_goals,
            )
        )
        self._last_match[home_key] = fixture.kickoff_utc
        self._last_match[away_key] = fixture.kickoff_utc

        home_rating = self._rating(league, fixture.home_team_id)
        away_rating = self._rating(league, fixture.away_team_id)
        expected_home = 1.0 / (
            1.0 + math.pow(10.0, -((home_rating + self.home_advantage) - away_rating) / 400.0)
        )
        actual_home = 1.0 if fixture.home_goals > fixture.away_goals else (
            0.5 if fixture.home_goals == fixture.away_goals else 0.0
        )
        change = self.elo_k * (actual_home - expected_home)
        self._elo[home_key] = home_rating + change
        self._elo[away_key] = away_rating - change
