#!/usr/bin/env python3
"""Freeze or verify the aggregate-only EPL same-match model benchmark.

The default command is a fast immutable-byte check. ``--rebuild`` reruns the
benchmark against ten SHA-pinned Football-Data.co.uk files and compares the
result with the committed release. ``--write`` performs that same rebuild and
writes a new immutable v1 release. No fixture row, scoreline, team name, odds
triplet, fitted estimator or fixture-level probability is published.

The four evaluation folds are entire EPL seasons (2022/23 through 2025/26).
Every fitted mapping uses complete earlier seasons only. Feature snapshots are
made 24 hours before kick-off and accept a final result only after a separate
24-hour availability lag. The LightGBM calibrator is fitted on expanding,
earlier-season out-of-fold raw probabilities. Closing odds are deliberately
reported as a later-information reference, never as a like-for-like 24-hour
model input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import sys
import urllib.request
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
VERSION = "epl-football-prediction-model-benchmark/1.0.0"
RELEASE_DATE = "2026-07-18"
SOURCE_MANIFEST_PATH = ROOT / "data/football-1x2-empirical-benchmark-1.0.0-manifest.json"
JSON_PATH = ROOT / "data/epl-football-prediction-model-benchmark-1.0.0.json"
CSV_PATH = ROOT / "data/epl-football-prediction-model-benchmark-1.0.0.csv"
MANIFEST_PATH = ROOT / "data/epl-football-prediction-model-benchmark-1.0.0-manifest.json"
GENERATOR_PATH = Path(__file__).resolve()
FEATURES_PATH = ROOT / "model/proofxi_ml/features.py"
TRAINING_PATH = ROOT / "model/proofxi_ml/training.py"
PUBLIC_JSON_PATH = "/research/epl-football-prediction-model-benchmark/1.0.0.json"
PUBLIC_CSV_PATH = "/downloads/epl-football-prediction-model-benchmark-1.0.0.csv"
PUBLIC_MANIFEST_PATH = "/research/epl-football-prediction-model-benchmark/1.0.0-manifest.json"
PUBLIC_GENERATOR_PATH = "/downloads/generate-epl-football-prediction-model-benchmark-1.0.0.py"
EVALUATION_SEASONS = ("2022/23", "2023/24", "2024/25", "2025/26")
CALIBRATION_OOF_START_SEASON = "2018/19"
EXPECTED_SOURCE_MATCHES = 3_800
EXPECTED_EVALUATION_MATCHES = 1_520
BOOTSTRAP_REPLICATES = 5_000
RANDOM_SEED = 20260718
TEAM_RIDGE = 1.0
GLOBAL_NUMERIC_RIDGE = 1e-8
IRLS_TOLERANCE = 1e-9
IRLS_MAX_ITERATIONS = 50
MAX_ABSOLUTE_ETA = 20.0
RHO_OUTER_BOUND = 0.25
TAU_FLOOR = 1e-10
AGGREGATE_LICENSE = "https://creativecommons.org/licenses/by/4.0/"
AGGREGATE_RIGHTS_SCOPE = (
    "CC BY 4.0 covers only Football Proof AI's original aggregate benchmark outputs and "
    "release metadata. It does not license or redistribute Football-Data.co.uk source rows "
    "or grant rights in the publisher's source files."
)
PACKAGE_VERSIONS = {
    "lightgbm": "4.6.0",
    "numpy": "2.2.6",
    "scikit-learn": "1.7.2",
    "scipy": "1.18.0",
}
MODEL_ORDER = (
    "uniform",
    "expanding_league_prior",
    "elo_multinomial_logit",
    "multinomial_logistic_14_feature",
    "ridge_poisson",
    "sequential_dixon_coles",
    "lightgbm_raw",
    "lightgbm_isotonic",
    "market_closing_proportional",
    "market_closing_shin",
)
HISTORY_MODEL_IDS = frozenset(
    {
        "expanding_league_prior",
        "elo_multinomial_logit",
        "multinomial_logistic_14_feature",
        "ridge_poisson",
        "sequential_dixon_coles",
        "lightgbm_raw",
        "lightgbm_isotonic",
    }
)
OUTCOME_LABELS = ("H", "D", "A")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def rounded(value: float | int | None, digits: int = 10) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"Cannot publish non-finite number {numeric}.")
    return round(numeric, digits)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate an empty mean.")
    return math.fsum(values) / len(values)


def iso_date(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()


def iso_week_start(timestamp: int) -> str:
    value = datetime.fromtimestamp(timestamp, timezone.utc)
    monday = value.date() - timedelta(days=value.weekday())
    return monday.isoformat()


def parse_timestamp(date_value: str, time_value: str | None) -> int:
    parts = [int(value) for value in date_value.strip().split("/")]
    if len(parts) != 3:
        raise ValueError(f"Invalid source date {date_value!r}.")
    day, month, raw_year = parts
    year = raw_year + 2000 if raw_year < 100 else raw_year
    clock = (time_value or "12:00").strip() or "12:00"
    clock_parts = [int(value) for value in clock.split(":")[:2]]
    if len(clock_parts) != 2:
        raise ValueError(f"Invalid source time {clock!r}.")
    hour, minute = clock_parts
    return int(datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp())


def outcome_index(label: str) -> int:
    try:
        return OUTCOME_LABELS.index(label)
    except ValueError as exc:
        raise ValueError(f"Invalid 1X2 outcome {label!r}.") from exc


def fixture_key(match: "Match") -> str:
    return f"{match.season}|{match.timestamp}|{match.home_team}|{match.away_team}"


def fixture_key_set_sha(keys: Iterable[str]) -> str:
    ordered = sorted(keys)
    return sha256(("\n".join(ordered) + "\n").encode("utf-8"))


@dataclass(frozen=True, slots=True)
class Match:
    fixture_id: int
    season: str
    season_index: int
    season_start: int
    timestamp: int
    source_order: int
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    target: int
    closing_odds: tuple[float, float, float] | None

    @property
    def key(self) -> str:
        return fixture_key(self)

    @property
    def block(self) -> str:
        return iso_week_start(self.timestamp)


@dataclass(frozen=True, slots=True)
class SourceInput:
    season: str
    role: str
    url: str
    local_relative_path: str
    sha256: str
    completed_match_rows: int


@dataclass(slots=True)
class GoalModel:
    beta: Any
    teams: tuple[str, ...]
    team_index: dict[str, int]
    convergence_iterations: int
    objective: float

    @property
    def attack_offset(self) -> int:
        return 2

    @property
    def defence_offset(self) -> int:
        return 2 + len(self.teams)


def source_contract() -> tuple[list[SourceInput], str]:
    source_bytes = SOURCE_MANIFEST_PATH.read_bytes()
    manifest = json.loads(source_bytes)
    entries = manifest.get("sourceInputs")
    if not isinstance(entries, list) or len(entries) != 10:
        raise RuntimeError("The source manifest must declare exactly ten EPL inputs.")
    selected: list[SourceInput] = []
    for index, entry in enumerate(entries):
        local_path = str(entry["url"]).split("/mmz4281/", 1)[1]
        selected.append(
            SourceInput(
                season=str(entry["season"]),
                role="evaluation" if str(entry["season"]) in EVALUATION_SEASONS else "prior-history",
                url=str(entry["url"]),
                local_relative_path=local_path,
                sha256=str(entry["sha256"]),
                completed_match_rows=int(entry["completedMatchRows"]),
            )
        )
        if index and selected[index - 1].season >= selected[index].season:
            raise RuntimeError("Source seasons are not chronological.")
    return selected, sha256(source_bytes)


def read_source(entry: SourceInput, source_dir: Path | None) -> bytes:
    if source_dir is not None:
        path = source_dir / entry.local_relative_path
        try:
            value = path.read_bytes()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing local source {path}; expected --source-dir/{entry.local_relative_path}."
            ) from exc
    else:
        request = urllib.request.Request(
            entry.url,
            headers={
                "User-Agent": (
                    "FootballProofAI-Same-Match-Model-Benchmark/1.0 "
                    "(+https://footballproofai.com/data-sources)"
                )
            },
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            value = response.read()
    digest = sha256(value)
    if digest != entry.sha256:
        raise RuntimeError(
            f"Source hash mismatch for {entry.season}: expected {entry.sha256}, got {digest}. "
            "Immutable v1 refuses changed publisher bytes."
        )
    return value


def decimal_odds(value: str | None) -> float | None:
    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 1.0 else None


def load_matches(source_dir: Path | None) -> tuple[list[Match], list[dict[str, Any]], str]:
    source_inputs, source_manifest_sha = source_contract()
    staged: list[tuple[int, SourceInput, int, dict[str, str]]] = []
    source_metadata: list[dict[str, Any]] = []
    for season_index, entry in enumerate(source_inputs):
        source_bytes = read_source(entry, source_dir)
        reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig")))
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if entry.role == "evaluation":
            required.update({"AvgCH", "AvgCD", "AvgCA"})
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{entry.season} source is missing fields {sorted(missing)!r}.")
        completed = 0
        for source_order, row in enumerate(reader, start=2):
            if row.get("FTR") not in OUTCOME_LABELS:
                continue
            completed += 1
            staged.append((season_index, entry, source_order, row))
        if completed != entry.completed_match_rows:
            raise RuntimeError(
                f"{entry.season} has {completed} completed matches; expected {entry.completed_match_rows}."
            )
        fields_used = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        if entry.role == "evaluation":
            fields_used.extend(["AvgCH", "AvgCD", "AvgCA"])
        source_metadata.append(
            {
                "season": entry.season,
                "role": entry.role,
                "url": entry.url,
                "localRelativePath": entry.local_relative_path,
                "sha256": entry.sha256,
                "completedMatchRows": completed,
                "fieldsUsed": fields_used,
            }
        )

    matches: list[Match] = []
    for fixture_id, (season_index, entry, source_order, row) in enumerate(staged, start=1):
        home_team = (row.get("HomeTeam") or "").strip()
        away_team = (row.get("AwayTeam") or "").strip()
        if not home_team or not away_team or home_team == away_team:
            raise RuntimeError(f"Invalid teams in {entry.season} source row {source_order}.")
        home_goals = int(row["FTHG"])
        away_goals = int(row["FTAG"])
        if min(home_goals, away_goals) < 0:
            raise RuntimeError(f"Invalid score in {entry.season} source row {source_order}.")
        closing_odds: tuple[float, float, float] | None = None
        if entry.role == "evaluation":
            odds = tuple(decimal_odds(row.get(field)) for field in ("AvgCH", "AvgCD", "AvgCA"))
            if any(value is None for value in odds):
                raise RuntimeError(
                    f"{entry.season} source row {source_order} lacks complete closing-average odds."
                )
            closing_odds = tuple(float(value) for value in odds)  # type: ignore[arg-type]
        matches.append(
            Match(
                fixture_id=fixture_id,
                season=entry.season,
                season_index=season_index,
                season_start=int(entry.season.split("/", 1)[0]),
                timestamp=parse_timestamp(row["Date"], row.get("Time")),
                source_order=source_order,
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                target=outcome_index(str(row["FTR"])),
                closing_odds=closing_odds,
            )
        )
    if len(matches) != EXPECTED_SOURCE_MATCHES:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_MATCHES} source matches, got {len(matches)}.")
    keys = [match.key for match in matches]
    if len(set(keys)) != len(keys):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise RuntimeError(f"Source contains duplicate fixture keys: {duplicates[:3]!r}.")
    matches.sort(key=lambda item: (item.timestamp, item.home_team, item.away_team, item.source_order))
    return matches, source_metadata, source_manifest_sha


def assert_runtime_versions() -> None:
    for package, expected in PACKAGE_VERSIONS.items():
        observed = importlib.metadata.version(package)
        if observed != expected:
            raise RuntimeError(f"{package} must be {expected} for immutable v1; found {observed}.")


def normalize_probabilities(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("A 1X2 forecast must contain exactly three probabilities.")
    clipped = [max(1e-12, float(value)) for value in values]
    if any(not math.isfinite(value) for value in clipped):
        raise ValueError("A 1X2 forecast contains a non-finite probability.")
    total = math.fsum(clipped)
    if not total > 0:
        raise ValueError("A 1X2 probability total must be positive.")
    normalized = tuple(value / total for value in clipped)
    if abs(math.fsum(normalized) - 1.0) > 1e-12:
        raise ValueError("A normalized 1X2 forecast does not sum to one.")
    return normalized  # type: ignore[return-value]


def proportional_market(odds: Sequence[float]) -> tuple[float, float, float]:
    implied = [1.0 / value for value in odds]
    return normalize_probabilities(implied)


def shin_market(odds: Sequence[float]) -> tuple[tuple[float, float, float], float]:
    implied = [1.0 / value for value in odds]
    booksum = math.fsum(implied)
    if abs(booksum - 1.0) <= 1e-12:
        return proportional_market(odds), 0.0
    if booksum <= 1.0:
        raise ValueError(f"Shin is undefined for underround booksum {booksum}.")
    coefficients = [value * value / booksum for value in implied]

    def probabilities_at(z: float) -> list[float]:
        return [
            (2.0 * value) / (math.sqrt(z * z + 4.0 * (1.0 - z) * value) + z)
            for value in coefficients
        ]

    def residual_at(z: float) -> float:
        return math.fsum(probabilities_at(z)) - 1.0

    lower = 0.0
    upper = 1.0 - sys.float_info.epsilon
    if not (residual_at(lower) > 0.0 and residual_at(upper) < 0.0):
        raise ValueError(f"Shin solver has no bracket for booksum {booksum}.")
    z = math.nan
    for _ in range(256):
        midpoint = (lower + upper) / 2.0
        residual = residual_at(midpoint)
        if abs(residual) <= 1e-13 or upper - lower <= 1e-14:
            z = midpoint
            break
        if residual > 0:
            lower = midpoint
        else:
            upper = midpoint
    if not math.isfinite(z):
        raise RuntimeError("Shin solver did not converge.")
    return normalize_probabilities(probabilities_at(z)), z


def build_feature_examples(matches: Sequence[Match]) -> tuple[dict[int, Any], tuple[str, ...], str]:
    sys.path.insert(0, str(ROOT / "model"))
    from proofxi_ml.domain import Fixture
    from proofxi_ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureEngine

    team_names = sorted({team for match in matches for team in (match.home_team, match.away_team)})
    team_ids = {team: index for index, team in enumerate(team_names, start=1)}
    fixtures = [
        Fixture(
            fixture_id=match.fixture_id,
            league_id=39,
            season=match.season_start,
            kickoff_utc=match.timestamp,
            home_team_id=team_ids[match.home_team],
            away_team_id=team_ids[match.away_team],
            home_name=match.home_team,
            away_name=match.away_team,
            status="FT",
            home_goals=match.home_goals,
            away_goals=match.away_goals,
        )
        for match in matches
    ]
    examples, _ = FeatureEngine().build(fixtures)
    by_fixture = {example.fixture.fixture_id: example for example in examples}
    if len(by_fixture) != len(matches):
        raise RuntimeError("Feature engine did not return exact fixture coverage.")
    for match in matches:
        example = by_fixture[match.fixture_id]
        if example.target != match.target:
            raise RuntimeError(f"Feature target mismatch for fixture {match.fixture_id}.")
        if example.snapshot.as_of_utc != match.timestamp - 86_400:
            raise RuntimeError(f"Feature as-of timestamp drift for fixture {match.fixture_id}.")
        if example.snapshot.source_max_match_utc >= example.snapshot.as_of_utc:
            raise RuntimeError(f"Feature leakage guard failed for fixture {match.fixture_id}.")
    return by_fixture, tuple(FEATURE_NAMES), FEATURE_SCHEMA_VERSION


def matrix_for(examples: Sequence[Any], feature_names: Sequence[str], np: Any) -> Any:
    return np.asarray(
        [
            [
                np.nan if example.snapshot.values[name] is None else example.snapshot.values[name]
                for name in feature_names
            ]
            for example in examples
        ],
        dtype=float,
    )


def fit_logistic_probabilities(
    train_examples: Sequence[Any],
    test_examples: Sequence[Any],
    feature_names: Sequence[str],
    np: Any,
) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2_000,
                    tol=1e-8,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    x_train = matrix_for(train_examples, feature_names, np)
    y_train = np.asarray([example.target for example in train_examples], dtype=int)
    x_test = matrix_for(test_examples, feature_names, np)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        pipeline.fit(x_train, y_train)
    classes = tuple(int(value) for value in pipeline.named_steps["logit"].classes_)
    if classes != (0, 1, 2):
        raise RuntimeError(f"Logistic class order drifted: {classes!r}.")
    return pipeline.predict_proba(x_test)


def build_expanding_prior(
    matches: Sequence[Match], evaluation: Sequence[Match]
) -> dict[str, tuple[float, float, float]]:
    evaluation_keys = {match.key for match in evaluation}
    ordered = sorted(matches, key=lambda item: (item.timestamp, item.home_team, item.away_team))
    available = sorted(
        ((match.timestamp + 86_400, match.target) for match in ordered),
        key=lambda item: item[0],
    )
    counts = [0, 0, 0]
    cursor = 0
    output: dict[str, tuple[float, float, float]] = {}
    for match in ordered:
        as_of = match.timestamp - 86_400
        while cursor < len(available) and available[cursor][0] < as_of:
            counts[available[cursor][1]] += 1
            cursor += 1
        if match.key in evaluation_keys:
            total = sum(counts)
            if total <= 0:
                raise RuntimeError("Expanding prior reached evaluation without eligible history.")
            output[match.key] = normalize_probabilities([value / total for value in counts])
    return output


def poisson_objective(beta: Any, matches: Sequence[Match], layout: GoalModel, np: Any) -> float:
    objective = 0.0
    for match in matches:
        for home in (True, False):
            attack_team = match.home_team if home else match.away_team
            defence_team = match.away_team if home else match.home_team
            indices = [
                0,
                *([1] if home else []),
                layout.attack_offset + layout.team_index[attack_team],
                layout.defence_offset + layout.team_index[defence_team],
            ]
            values = [1.0, *([1.0] if home else []), 1.0, -1.0]
            eta = math.fsum(float(beta[index]) * value for index, value in zip(indices, values))
            if abs(eta) > MAX_ABSOLUTE_ETA:
                return math.inf
            mu = math.exp(eta)
            target = match.home_goals if home else match.away_goals
            objective += mu - target * eta
    penalties = np.full(len(beta), TEAM_RIDGE, dtype=float)
    penalties[:2] = GLOBAL_NUMERIC_RIDGE
    objective += 0.5 * float(np.dot(penalties, beta * beta))
    return objective


def fit_poisson_goal_model(matches: Sequence[Match], np: Any) -> GoalModel:
    if len(matches) < 380:
        raise RuntimeError(f"Poisson training has only {len(matches)} matches.")
    teams = tuple(sorted({team for match in matches for team in (match.home_team, match.away_team)}))
    team_index = {team: index for index, team in enumerate(teams)}
    parameter_count = 2 + 2 * len(teams)
    home_mean = mean([float(match.home_goals) for match in matches])
    away_mean = mean([float(match.away_goals) for match in matches])
    beta = np.zeros(parameter_count, dtype=float)
    beta[0] = math.log(away_mean)
    beta[1] = math.log(home_mean / away_mean)
    layout = GoalModel(beta, teams, team_index, 0, math.nan)
    objective = poisson_objective(beta, matches, layout, np)

    for iteration in range(1, IRLS_MAX_ITERATIONS + 1):
        gradient = np.zeros(parameter_count, dtype=float)
        hessian = np.zeros((parameter_count, parameter_count), dtype=float)
        for match in matches:
            for home in (True, False):
                attack_team = match.home_team if home else match.away_team
                defence_team = match.away_team if home else match.home_team
                indices = np.asarray(
                    [
                        0,
                        *([1] if home else []),
                        layout.attack_offset + team_index[attack_team],
                        layout.defence_offset + team_index[defence_team],
                    ],
                    dtype=int,
                )
                values = np.asarray([1.0, *([1.0] if home else []), 1.0, -1.0])
                eta = float(np.dot(beta[indices], values))
                if abs(eta) > MAX_ABSOLUTE_ETA:
                    raise RuntimeError(f"Poisson IRLS produced unsafe eta {eta}.")
                mu = math.exp(eta)
                target = match.home_goals if home else match.away_goals
                gradient[indices] += values * (mu - target)
                hessian[np.ix_(indices, indices)] += mu * np.outer(values, values)
        penalties = np.full(parameter_count, TEAM_RIDGE, dtype=float)
        penalties[:2] = GLOBAL_NUMERIC_RIDGE
        gradient += penalties * beta
        hessian[np.diag_indices(parameter_count)] += penalties
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        candidate = None
        candidate_objective = math.inf
        while scale >= 2.0**-20:
            attempted = beta - scale * step
            attempted_layout = GoalModel(attempted, teams, team_index, iteration, math.nan)
            attempted_objective = poisson_objective(attempted, matches, attempted_layout, np)
            if math.isfinite(attempted_objective) and attempted_objective <= objective + 1e-10:
                candidate = attempted
                candidate_objective = attempted_objective
                break
            scale /= 2.0
        if candidate is None:
            raise RuntimeError(f"Poisson IRLS line search failed at iteration {iteration}.")
        maximum_step = float(np.max(np.abs(candidate - beta)))
        beta = candidate
        objective = candidate_objective
        layout = GoalModel(beta, teams, team_index, iteration, objective)
        if maximum_step <= IRLS_TOLERANCE:
            return layout
    raise RuntimeError(f"Poisson IRLS did not converge within {IRLS_MAX_ITERATIONS} iterations.")


def predict_goal_rates(model: GoalModel, home_team: str, away_team: str) -> tuple[float, float, bool]:
    def effect(team: str, offset: int) -> float:
        index = model.team_index.get(team)
        return 0.0 if index is None else float(model.beta[offset + index])

    home_eta = (
        float(model.beta[0])
        + float(model.beta[1])
        + effect(home_team, model.attack_offset)
        - effect(away_team, model.defence_offset)
    )
    away_eta = (
        float(model.beta[0])
        + effect(away_team, model.attack_offset)
        - effect(home_team, model.defence_offset)
    )
    home_lambda = math.exp(home_eta)
    away_lambda = math.exp(away_eta)
    if not (0 < home_lambda <= 20 and 0 < away_lambda <= 20):
        raise RuntimeError(f"Unsafe Poisson rates {home_lambda}/{away_lambda}.")
    return home_lambda, away_lambda, home_team not in model.team_index or away_team not in model.team_index


def rho_coefficient(home_goals: int, away_goals: int, home_lambda: float, away_lambda: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return -home_lambda * away_lambda
    if home_goals == 0 and away_goals == 1:
        return home_lambda
    if home_goals == 1 and away_goals == 0:
        return away_lambda
    if home_goals == 1 and away_goals == 1:
        return -1.0
    return 0.0


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_lambda: float,
    away_lambda: float,
    rho: float,
) -> float:
    tau = 1.0 + rho_coefficient(home_goals, away_goals, home_lambda, away_lambda) * rho
    if not math.isfinite(tau) or tau <= TAU_FLOOR:
        raise RuntimeError(f"Invalid Dixon-Coles tau {tau}.")
    return tau


def estimate_dixon_coles_rho(matches: Sequence[Match], model: GoalModel) -> tuple[float, int]:
    observations = []
    lower = -RHO_OUTER_BOUND
    upper = RHO_OUTER_BOUND
    for match in matches:
        home_lambda, away_lambda, _ = predict_goal_rates(model, match.home_team, match.away_team)
        coefficient = rho_coefficient(match.home_goals, match.away_goals, home_lambda, away_lambda)
        if coefficient:
            observations.append(coefficient)
        lower = max(lower, -1.0 / home_lambda + TAU_FLOOR, -1.0 / away_lambda + TAU_FLOOR)
        upper = min(
            upper,
            1.0 / (home_lambda * away_lambda) - TAU_FLOOR,
            1.0 - TAU_FLOOR,
        )
    if len(observations) < 25 or not lower < upper:
        raise RuntimeError("Dixon-Coles rho does not have a valid training interval.")

    def score(rho: float) -> float:
        return math.fsum(value / (1.0 + value * rho) for value in observations)

    epsilon = max(1e-10, (upper - lower) * 1e-10)
    left = lower + epsilon
    right = upper - epsilon
    if not (score(left) > 0 and score(right) < 0):
        raise RuntimeError("Dixon-Coles rho optimum lies on a feasibility boundary.")
    for _ in range(160):
        midpoint = (left + right) / 2.0
        if score(midpoint) > 0:
            left = midpoint
        else:
            right = midpoint
    rho = (left + right) / 2.0
    if abs(score(rho)) > 1e-8:
        raise RuntimeError("Dixon-Coles rho bisection did not converge.")
    return rho, len(observations)


def poisson_mass(value: float) -> list[float]:
    probabilities = [math.exp(-value)]
    cumulative = probabilities[0]
    for goals in range(1, 61):
        probabilities.append(probabilities[-1] * value / goals)
        cumulative += probabilities[-1]
        if goals >= 8 and 1.0 - cumulative <= 1e-13:
            break
    if 1.0 - cumulative > 1e-12:
        raise RuntimeError(f"Poisson tail did not close for lambda {value}.")
    return probabilities


def score_matrix_probabilities(
    home_lambda: float, away_lambda: float, rho: float
) -> tuple[float, float, float]:
    home_mass = poisson_mass(home_lambda)
    away_mass = poisson_mass(away_lambda)
    outcomes = [0.0, 0.0, 0.0]
    total = 0.0
    for home_goals, home_probability in enumerate(home_mass):
        for away_goals, away_probability in enumerate(away_mass):
            probability = (
                home_probability
                * away_probability
                * dixon_coles_tau(home_goals, away_goals, home_lambda, away_lambda, rho)
            )
            total += probability
            outcome = 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2
            outcomes[outcome] += probability
    if abs(total - 1.0) > 1e-10:
        raise RuntimeError(f"Poisson score matrix mass is {total}.")
    return normalize_probabilities(outcomes)


def metric_rows(
    matches: Sequence[Match], predictions: dict[str, tuple[float, float, float]]
) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        probability = predictions[match.key]
        target = match.target
        brier = math.fsum(
            (value - (1.0 if index == target else 0.0)) ** 2
            for index, value in enumerate(probability)
        ) / 3.0
        log_loss = -math.log(max(1e-15, probability[target]))
        observed_cumulative = (1.0 if target == 0 else 0.0, 1.0 if target <= 1 else 0.0)
        predicted_cumulative = (probability[0], probability[0] + probability[1])
        rps = math.fsum(
            (predicted - observed) ** 2
            for predicted, observed in zip(predicted_cumulative, observed_cumulative)
        ) / 2.0
        maximum = max(probability)
        top = [index for index, value in enumerate(probability) if abs(value - maximum) <= 1e-12]
        fractional_hit = (1.0 / len(top)) if target in top else 0.0
        rows.append(
            {
                "key": match.key,
                "season": match.season,
                "block": match.block,
                "target": target,
                "probabilities": probability,
                "brierClassAverage": brier,
                "logLossNats": log_loss,
                "normalizedRps": rps,
                "topConfidence": maximum,
                "topPickFractionalHit": fractional_hit,
            }
        )
    return rows


def calibration_summary(rows: Sequence[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    top_bins = []
    top_ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            row
            for row in rows
            if lower <= row["topConfidence"] < upper
            or (index == bins - 1 and row["topConfidence"] == 1.0)
        ]
        if selected:
            mean_confidence = mean([row["topConfidence"] for row in selected])
            observed = mean([row["topPickFractionalHit"] for row in selected])
            absolute_gap = abs(observed - mean_confidence)
            top_ece += len(selected) / len(rows) * absolute_gap
        else:
            mean_confidence = None
            observed = None
            absolute_gap = None
        top_bins.append(
            {
                "lower": rounded(lower),
                "upper": rounded(upper),
                "n": len(selected),
                "meanConfidence": rounded(mean_confidence),
                "observedHitRateFractionalTies": rounded(observed),
                "absoluteGap": rounded(absolute_gap),
            }
        )

    class_ece = []
    for class_index, label in enumerate(OUTCOME_LABELS):
        error = 0.0
        for bin_index in range(bins):
            lower = bin_index / bins
            upper = (bin_index + 1) / bins
            selected = [
                row
                for row in rows
                if lower <= row["probabilities"][class_index] < upper
                or (bin_index == bins - 1 and row["probabilities"][class_index] == 1.0)
            ]
            if selected:
                predicted = mean([row["probabilities"][class_index] for row in selected])
                observed = mean([1.0 if row["target"] == class_index else 0.0 for row in selected])
                error += len(selected) / len(rows) * abs(predicted - observed)
        class_ece.append({"outcome": label, "ece10": rounded(error)})
    return {
        "topLabelEce10": top_ece,
        "macroClasswiseEce10": mean([float(entry["ece10"]) for entry in class_ece]),
        "classwise": class_ece,
        "topLabelBins": top_bins,
    }


def aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    calibration = calibration_summary(rows)
    return {
        "brierClassAverage": rounded(mean([row["brierClassAverage"] for row in rows])),
        "logLossNats": rounded(mean([row["logLossNats"] for row in rows])),
        "normalizedRps": rounded(mean([row["normalizedRps"] for row in rows])),
        "topPickHitRateFractionalTies": rounded(
            mean([row["topPickFractionalHit"] for row in rows])
        ),
        "topLabelEce10": rounded(float(calibration["topLabelEce10"])),
        "macroClasswiseEce10": rounded(float(calibration["macroClasswiseEce10"])),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def deterministic_random(seed_label: str) -> Callable[[], float]:
    state = int.from_bytes(hashlib.sha256(f"{VERSION}:{seed_label}".encode()).digest()[:4], "little") or 1

    def random() -> float:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        return state / 0x1_0000_0000

    return random


def paired_week_bootstrap(
    model_rows: Sequence[dict[str, Any]],
    reference_rows: Sequence[dict[str, Any]],
    metric: str,
    seed_label: str,
) -> dict[str, Any]:
    reference_by_key = {row["key"]: row for row in reference_rows}
    by_block: dict[str, list[float]] = defaultdict(list)
    for row in model_rows:
        by_block[row["block"]].append(row[metric] - reference_by_key[row["key"]][metric])
    blocks = [by_block[key] for key in sorted(by_block)]
    if len(blocks) < 20:
        raise RuntimeError(f"Paired bootstrap has only {len(blocks)} ISO-week blocks.")
    random = deterministic_random(seed_label)
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        n = 0
        for _ in blocks:
            sampled = blocks[min(len(blocks) - 1, int(random() * len(blocks)))]
            total += math.fsum(sampled)
            n += len(sampled)
        estimates.append(total / n)
    point = mean([value for block in blocks for value in block])
    return {
        "pointDifferenceModelMinusShin": rounded(point),
        "ci95": [rounded(percentile(estimates, 0.025)), rounded(percentile(estimates, 0.975))],
    }


def model_definitions(feature_names: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": "uniform",
            "label": "Uniform 1/3 baseline",
            "category": "baseline",
            "informationSet": "fixed_distribution",
            "method": "Fixed probabilities (1/3, 1/3, 1/3) for every match.",
            "fitSchedule": "Never fitted.",
            "inputs": [],
        },
        {
            "id": "expanding_league_prior",
            "label": "Availability-safe expanding EPL prior",
            "category": "baseline",
            "informationSet": "past_results_24h_safe",
            "method": "Observed H/D/A frequencies from finals whose 24-hour result lag ended strictly before the 24-hour forecast timestamp.",
            "fitSchedule": "Updated before each fixture timestamp without same-time or unavailable results.",
            "inputs": ["past EPL outcomes"],
        },
        {
            "id": "elo_multinomial_logit",
            "label": "Elo to multinomial logistic regression",
            "category": "history_only_model",
            "informationSet": "history_features_24h_safe",
            "method": "Three-class L2 multinomial logistic regression on the locked home-advantage Elo difference.",
            "fitSchedule": "Refitted once before each evaluation season using complete earlier seasons only.",
            "inputs": ["elo_diff_home_adv"],
        },
        {
            "id": "multinomial_logistic_14_feature",
            "label": "14-feature multinomial logistic regression",
            "category": "history_only_model",
            "informationSet": "history_features_24h_safe",
            "method": "Median imputation, standardization and L2 three-class multinomial logistic regression.",
            "fitSchedule": "Refitted once before each evaluation season using complete earlier seasons only.",
            "inputs": list(feature_names),
        },
        {
            "id": "ridge_poisson",
            "label": "Independent ridge-Poisson",
            "category": "history_only_model",
            "informationSet": "past_results_24h_safe",
            "method": "Independent home/away Poisson goal rates with home advantage and ridge attack/defence effects, converted to 1X2 probabilities.",
            "fitSchedule": "Refitted once before each evaluation season using complete earlier seasons only.",
            "inputs": ["past EPL final scores", "home team", "away team"],
        },
        {
            "id": "sequential_dixon_coles",
            "label": "Sequential training-only Dixon-Coles",
            "category": "history_only_model",
            "informationSet": "past_results_24h_safe",
            "method": "Training-only Dixon-Coles low-score rho applied to the identical ridge-Poisson rates; not a joint Dixon-Coles fit.",
            "fitSchedule": "Poisson rates and rho are refitted once before each evaluation season from complete earlier seasons only.",
            "inputs": ["past EPL final scores", "home team", "away team"],
        },
        {
            "id": "lightgbm_raw",
            "label": "Raw 14-feature LightGBM",
            "category": "ablation",
            "informationSet": "history_features_24h_safe",
            "method": "Locked deterministic multiclass LightGBM before probability calibration.",
            "fitSchedule": "Refitted once before each season on all complete earlier seasons; published as a calibration ablation.",
            "inputs": list(feature_names),
        },
        {
            "id": "lightgbm_isotonic",
            "label": "LightGBM plus training-only isotonic calibration",
            "category": "history_only_model",
            "informationSet": "history_features_24h_safe",
            "method": "The same raw LightGBM probabilities passed through one-vs-rest isotonic calibrators and renormalized.",
            "fitSchedule": "The evaluation raw model uses all earlier seasons; calibrators use expanding raw predictions generated out of fold on earlier seasons only.",
            "inputs": list(feature_names),
        },
        {
            "id": "market_closing_proportional",
            "label": "Proportional de-vig closing market",
            "category": "later_information_reference",
            "informationSet": "closing_market_later_information",
            "method": "Average closing 1X2 decimal odds converted to implied weights and normalized by booksum.",
            "fitSchedule": "Not a fitted AI model; closing prices are observed later than the 24-hour model timestamp.",
            "inputs": ["AvgCH", "AvgCD", "AvgCA"],
        },
        {
            "id": "market_closing_shin",
            "label": "Shin de-vig closing market",
            "category": "later_information_reference",
            "informationSet": "closing_market_later_information",
            "method": "Average closing 1X2 decimal odds de-vigged with deterministic Shin bisection.",
            "fitSchedule": "Not a fitted AI model; closing prices are observed later than the 24-hour model timestamp.",
            "inputs": ["AvgCH", "AvgCD", "AvgCA"],
        },
    ]


def predictions_for_all_models(
    matches: Sequence[Match],
    evaluation: Sequence[Match],
    examples_by_fixture: dict[int, Any],
    feature_names: Sequence[str],
    np: Any,
) -> tuple[dict[str, dict[str, tuple[float, float, float]]], list[dict[str, Any]]]:
    predictions: dict[str, dict[str, tuple[float, float, float]]] = {
        model_id: {} for model_id in MODEL_ORDER
    }
    diagnostics: list[dict[str, Any]] = []
    for match in evaluation:
        predictions["uniform"][match.key] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        if match.closing_odds is None:
            raise RuntimeError(f"Evaluation fixture {match.key} has no closing odds.")
        predictions["market_closing_proportional"][match.key] = proportional_market(
            match.closing_odds
        )
        shin, _ = shin_market(match.closing_odds)
        predictions["market_closing_shin"][match.key] = shin
    predictions["expanding_league_prior"] = build_expanding_prior(matches, evaluation)

    by_season: dict[str, list[Match]] = defaultdict(list)
    for match in matches:
        by_season[match.season].append(match)
    seasons = [entry.season for entry in source_contract()[0]]
    examples_by_season = {
        season: [examples_by_fixture[match.fixture_id] for match in by_season[season]]
        for season in seasons
    }

    # Generate expanding season-level out-of-fold LightGBM probabilities once.
    from lightgbm import LGBMClassifier
    from proofxi_ml.training import (
        TrainingConfig,
        _fit_calibrators,
        _model,
        calibrate_probabilities,
        predict_raw_probabilities,
    )
    from sklearn.isotonic import IsotonicRegression

    config = TrainingConfig()
    lgbm_oof: dict[str, Any] = {}
    for target_index in range(seasons.index(CALIBRATION_OOF_START_SEASON), len(seasons)):
        season = seasons[target_index]
        train_examples = [
            example
            for earlier in seasons[:target_index]
            for example in examples_by_season[earlier]
        ]
        test_examples = examples_by_season[season]
        raw_model = _model(config, LGBMClassifier)
        raw_model.fit(
            matrix_for(train_examples, feature_names, np),
            np.asarray([example.target for example in train_examples], dtype=int),
        )
        raw = predict_raw_probabilities(raw_model, matrix_for(test_examples, feature_names, np))
        lgbm_oof[season] = raw

    elo_feature = ("elo_diff_home_adv",)
    for evaluation_season in EVALUATION_SEASONS:
        target_index = seasons.index(evaluation_season)
        train_matches = [match for match in matches if match.season_index < target_index]
        test_matches = sorted(
            by_season[evaluation_season],
            key=lambda item: (item.timestamp, item.home_team, item.away_team),
        )
        train_examples = [examples_by_fixture[match.fixture_id] for match in train_matches]
        test_examples = [examples_by_fixture[match.fixture_id] for match in test_matches]
        if len(test_matches) != 380:
            raise RuntimeError(f"{evaluation_season} evaluation does not contain 380 matches.")

        elo_probabilities = fit_logistic_probabilities(
            train_examples, test_examples, elo_feature, np
        )
        logistic_probabilities = fit_logistic_probabilities(
            train_examples, test_examples, feature_names, np
        )
        raw_lgbm = lgbm_oof[evaluation_season]
        calibration_seasons = seasons[
            seasons.index(CALIBRATION_OOF_START_SEASON) : target_index
        ]
        calibration_raw = np.vstack([lgbm_oof[season] for season in calibration_seasons])
        calibration_targets = np.asarray(
            [
                example.target
                for season in calibration_seasons
                for example in examples_by_season[season]
            ],
            dtype=int,
        )
        calibrators = _fit_calibrators(
            calibration_raw, calibration_targets, IsotonicRegression, np
        )
        calibrated_lgbm = calibrate_probabilities(raw_lgbm, calibrators, np)

        poisson_model = fit_poisson_goal_model(train_matches, np)
        rho, low_score_training_matches = estimate_dixon_coles_rho(train_matches, poisson_model)
        cold_starts = 0
        for row_index, match in enumerate(test_matches):
            predictions["elo_multinomial_logit"][match.key] = normalize_probabilities(
                elo_probabilities[row_index]
            )
            predictions["multinomial_logistic_14_feature"][match.key] = normalize_probabilities(
                logistic_probabilities[row_index]
            )
            predictions["lightgbm_raw"][match.key] = normalize_probabilities(raw_lgbm[row_index])
            predictions["lightgbm_isotonic"][match.key] = normalize_probabilities(
                calibrated_lgbm[row_index]
            )
            home_lambda, away_lambda, cold_start = predict_goal_rates(
                poisson_model, match.home_team, match.away_team
            )
            cold_starts += int(cold_start)
            predictions["ridge_poisson"][match.key] = score_matrix_probabilities(
                home_lambda, away_lambda, 0.0
            )
            predictions["sequential_dixon_coles"][match.key] = score_matrix_probabilities(
                home_lambda, away_lambda, rho
            )
        diagnostics.append(
            {
                "evaluationSeason": evaluation_season,
                "trainingSeasons": seasons[:target_index],
                "trainingMatchCount": len(train_matches),
                "calibrationOofSeasons": calibration_seasons,
                "calibrationOofMatchCount": len(calibration_targets),
                "poissonTeamCount": len(poisson_model.teams),
                "poissonConvergenceIterations": poisson_model.convergence_iterations,
                "dixonColesLowScoreTrainingMatches": low_score_training_matches,
                "dixonColesRho": rounded(rho),
                "poissonColdStartEvaluationMatches": cold_starts,
            }
        )
    return predictions, diagnostics


def csv_escape(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(character in text for character in (",", '"', "\r", "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


CSV_COLUMNS = (
    "recordType",
    "modelId",
    "modelLabel",
    "category",
    "informationSet",
    "season",
    "n",
    "weekCount",
    "coverageStart",
    "coverageEnd",
    "fixtureKeySetSha256",
    "brierClassAverage",
    "logLossNats",
    "normalizedRps",
    "topPickHitRateFractionalTies",
    "topLabelEce10",
    "macroClasswiseEce10",
    "confidenceLower",
    "confidenceUpper",
    "binN",
    "meanConfidence",
    "observedHitRateFractionalTies",
    "absoluteCalibrationGap",
    "referenceModelId",
    "metric",
    "pointDifferenceModelMinusShin",
    "ci95Lower",
    "ci95Upper",
)


def artifact_csv(artifact: dict[str, Any]) -> bytes:
    definitions = {entry["id"]: entry for entry in artifact["models"]}
    records: list[dict[str, Any]] = []
    for row in artifact["leaderboard"]:
        records.append(
            {
                "recordType": "leaderboard",
                "modelId": row["modelId"],
                "modelLabel": row["label"],
                "category": row["category"],
                "informationSet": row["informationSet"],
                "n": row["n"],
                "weekCount": artifact["evaluation"]["evaluationWeekCount"],
                "coverageStart": artifact["evaluation"]["temporalCoverageStart"],
                "coverageEnd": artifact["evaluation"]["temporalCoverageEnd"],
                "fixtureKeySetSha256": row["fixtureKeySetSha256"],
                **row["metrics"],
            }
        )
    for season in artifact["seasonSummaries"]:
        for row in season["models"]:
            definition = definitions[row["modelId"]]
            records.append(
                {
                    "recordType": "season_summary",
                    "modelId": row["modelId"],
                    "modelLabel": definition["label"],
                    "category": definition["category"],
                    "informationSet": definition["informationSet"],
                    "season": season["season"],
                    "n": season["n"],
                    "weekCount": season["weekCount"],
                    "coverageStart": season["coverageStart"],
                    "coverageEnd": season["coverageEnd"],
                    **row["metrics"],
                }
            )
    for entry in artifact["calibration"]:
        definition = definitions[entry["modelId"]]
        for bin_entry in entry["topLabelBins"]:
            records.append(
                {
                    "recordType": "calibration_bin",
                    "modelId": entry["modelId"],
                    "modelLabel": definition["label"],
                    "category": definition["category"],
                    "informationSet": definition["informationSet"],
                    "n": artifact["evaluation"]["evaluationMatchCount"],
                    "confidenceLower": bin_entry["lower"],
                    "confidenceUpper": bin_entry["upper"],
                    "binN": bin_entry["n"],
                    "meanConfidence": bin_entry["meanConfidence"],
                    "observedHitRateFractionalTies": bin_entry[
                        "observedHitRateFractionalTies"
                    ],
                    "absoluteCalibrationGap": bin_entry["absoluteGap"],
                }
            )
    for comparison in artifact["pairedUncertainty"]["comparisons"]:
        definition = definitions[comparison["modelId"]]
        for metric, result in comparison["metrics"].items():
            records.append(
                {
                    "recordType": "paired_bootstrap",
                    "modelId": comparison["modelId"],
                    "modelLabel": definition["label"],
                    "category": definition["category"],
                    "informationSet": definition["informationSet"],
                    "n": artifact["evaluation"]["evaluationMatchCount"],
                    "weekCount": artifact["pairedUncertainty"]["blockCount"],
                    "referenceModelId": artifact["pairedUncertainty"]["referenceModelId"],
                    "metric": metric,
                    "pointDifferenceModelMinusShin": result[
                        "pointDifferenceModelMinusShin"
                    ],
                    "ci95Lower": result["ci95"][0],
                    "ci95Upper": result["ci95"][1],
                }
            )
    lines = [",".join(CSV_COLUMNS)]
    lines.extend(
        ",".join(csv_escape(record.get(column)) for column in CSV_COLUMNS)
        for record in records
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_release(source_dir: Path | None) -> tuple[bytes, bytes, bytes]:
    assert_runtime_versions()
    import numpy as np

    matches, source_inputs, source_manifest_sha = load_matches(source_dir)
    evaluation = [match for match in matches if match.season in EVALUATION_SEASONS]
    evaluation.sort(key=lambda item: (item.timestamp, item.home_team, item.away_team))
    if len(evaluation) != EXPECTED_EVALUATION_MATCHES:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUATION_MATCHES} evaluation matches, got {len(evaluation)}."
        )
    evaluation_keys = {match.key for match in evaluation}
    evaluation_key_sha = fixture_key_set_sha(evaluation_keys)
    examples_by_fixture, feature_names, feature_schema_version = build_feature_examples(matches)
    predictions, fold_diagnostics = predictions_for_all_models(
        matches, evaluation, examples_by_fixture, feature_names, np
    )

    scored: dict[str, list[dict[str, Any]]] = {}
    for model_id in MODEL_ORDER:
        keys = set(predictions[model_id])
        if keys != evaluation_keys or len(predictions[model_id]) != EXPECTED_EVALUATION_MATCHES:
            missing = sorted(evaluation_keys - keys)[:3]
            extra = sorted(keys - evaluation_keys)[:3]
            raise RuntimeError(
                f"{model_id} fixture coverage mismatch: missing={missing!r}, extra={extra!r}."
            )
        if fixture_key_set_sha(keys) != evaluation_key_sha:
            raise RuntimeError(f"{model_id} fixture-key digest drifted.")
        scored[model_id] = metric_rows(evaluation, predictions[model_id])

    definitions = model_definitions(feature_names)
    definition_by_id = {entry["id"]: entry for entry in definitions}
    if tuple(definition_by_id) != MODEL_ORDER:
        raise RuntimeError("Model definitions do not match the locked model order.")
    leaderboard = []
    for model_id in MODEL_ORDER:
        definition = definition_by_id[model_id]
        leaderboard.append(
            {
                "modelId": model_id,
                "label": definition["label"],
                "category": definition["category"],
                "informationSet": definition["informationSet"],
                "n": len(scored[model_id]),
                "fixtureKeySetSha256": evaluation_key_sha,
                "metrics": aggregate_metrics(scored[model_id]),
            }
        )
    for rank, row in enumerate(
        sorted(leaderboard, key=lambda item: item["metrics"]["brierClassAverage"]), start=1
    ):
        row["overallBrierRankIncludingLaterMarket"] = rank
    history_rows = [row for row in leaderboard if row["modelId"] in HISTORY_MODEL_IDS]
    for rank, row in enumerate(
        sorted(history_rows, key=lambda item: item["metrics"]["brierClassAverage"]), start=1
    ):
        row["historyOnlyBrierRank"] = rank

    season_summaries = []
    for season in EVALUATION_SEASONS:
        season_matches = [match for match in evaluation if match.season == season]
        season_keys = {match.key for match in season_matches}
        season_summaries.append(
            {
                "season": season,
                "n": len(season_matches),
                "weekCount": len({match.block for match in season_matches}),
                "coverageStart": iso_date(min(match.timestamp for match in season_matches)),
                "coverageEnd": iso_date(max(match.timestamp for match in season_matches)),
                "fixtureKeySetSha256": fixture_key_set_sha(season_keys),
                "models": [
                    {
                        "modelId": model_id,
                        "metrics": aggregate_metrics(
                            [row for row in scored[model_id] if row["season"] == season]
                        ),
                    }
                    for model_id in MODEL_ORDER
                ],
            }
        )

    calibration = []
    for model_id in MODEL_ORDER:
        report = calibration_summary(scored[model_id])
        calibration.append(
            {
                "modelId": model_id,
                "topLabelEce10": rounded(float(report["topLabelEce10"])),
                "macroClasswiseEce10": rounded(float(report["macroClasswiseEce10"])),
                "classwise": report["classwise"],
                "topLabelBins": report["topLabelBins"],
            }
        )

    reference_rows = scored["market_closing_shin"]
    paired_comparisons = []
    for model_id in MODEL_ORDER:
        if model_id == "market_closing_shin":
            continue
        paired_comparisons.append(
            {
                "modelId": model_id,
                "metrics": {
                    metric: paired_week_bootstrap(
                        scored[model_id],
                        reference_rows,
                        metric,
                        f"{model_id}:{metric}:vs-shin",
                    )
                    for metric in ("brierClassAverage", "logLossNats", "normalizedRps")
                },
            }
        )
    raw_calibrated_ablation = {
        "rawModelId": "lightgbm_raw",
        "calibratedModelId": "lightgbm_isotonic",
        "interpretation": "Calibrated score minus raw score; negative favours training-only isotonic calibration.",
        "metrics": {},
    }
    for metric in ("brierClassAverage", "logLossNats", "normalizedRps"):
        # Reuse the paired bootstrap implementation by treating raw LightGBM as the reference.
        result = paired_week_bootstrap(
            scored["lightgbm_isotonic"],
            scored["lightgbm_raw"],
            metric,
            f"lightgbm-isotonic-vs-raw:{metric}",
        )
        raw_calibrated_ablation["metrics"][metric] = {
            "pointDifferenceCalibratedMinusRaw": result[
                "pointDifferenceModelMinusShin"
            ],
            "ci95": result["ci95"],
        }
    generator_sha = sha256(GENERATOR_PATH.read_bytes())
    dependency_files = [
        {
            "repositoryPath": str(FEATURES_PATH.relative_to(ROOT)),
            "sha256": sha256(FEATURES_PATH.read_bytes()),
        },
        {
            "repositoryPath": str(TRAINING_PATH.relative_to(ROOT)),
            "sha256": sha256(TRAINING_PATH.read_bytes()),
        },
    ]
    best_history = min(
        history_rows, key=lambda item: item["metrics"]["brierClassAverage"]
    )
    shin_row = next(row for row in leaderboard if row["modelId"] == "market_closing_shin")
    artifact: dict[str, Any] = {
        "schemaVersion": VERSION,
        "releaseDate": RELEASE_DATE,
        "title": "Same-match EPL football prediction model benchmark",
        "status": "aggregate research benchmark; no production model activation",
        "directAnswer": (
            f"On the same {EXPECTED_EVALUATION_MATCHES:,} EPL matches, {best_history['label']} had the "
            f"lowest history-only class-averaged Brier score ({best_history['metrics']['brierClassAverage']:.10f}). "
            f"The later-information Shin closing-market reference scored {shin_row['metrics']['brierClassAverage']:.10f}; "
            "that timing difference prevents a like-for-like algorithm-versus-market claim."
        ),
        "league": {"code": "E0", "name": "English Premier League", "country": "England"},
        "evaluation": {
            "sourceSeasonCount": len(source_inputs),
            "sourceMatchCount": len(matches),
            "trainingSeasons": [entry["season"] for entry in source_inputs if entry["role"] == "prior-history"],
            "evaluationSeasons": list(EVALUATION_SEASONS),
            "evaluationMatchCount": len(evaluation),
            "evaluationWeekCount": len({match.block for match in evaluation}),
            "temporalCoverageStart": iso_date(evaluation[0].timestamp),
            "temporalCoverageEnd": iso_date(evaluation[-1].timestamp),
            "evaluationFixtureKeySetSha256": evaluation_key_sha,
            "allModelsFullCoverage": True,
        },
        "protocol": {
            "name": "Four-fold complete-season expanding-origin evaluation",
            "foldUnit": "One complete EPL season.",
            "evaluationFolds": len(EVALUATION_SEASONS),
            "fittingCutoff": "Every learned mapping for an evaluation season uses only complete earlier seasons.",
            "featureAsOf": "24 hours before scheduled kick-off.",
            "resultAvailabilityLag": "24 hours after kick-off; eligible source timestamps must be strictly earlier than feature as-of.",
            "sameKickoffSafety": "Feature snapshots are batched by identical kick-off; no same-time result can enter another snapshot.",
            "lightgbmCalibration": (
                "One-vs-rest isotonic calibrators use expanding earlier-season raw probabilities that were themselves "
                "generated by models trained only before each calibration season. The evaluation raw model then fits all "
                "complete earlier seasons."
            ),
            "marketTiming": (
                "AvgCH/AvgCD/AvgCA are average closing odds and therefore a later-information contextual reference, "
                "not a same-information 24-hour baseline."
            ),
            "algorithmRankingBoundary": (
                "Rows are forecast systems with declared protocols and information sets. The table cannot isolate an "
                "algorithm-causal ranking, and the market rows cannot be compared as if observed 24 hours before kick-off."
            ),
            "foldDiagnostics": fold_diagnostics,
        },
        "scoring": {
            "primary": "Class-averaged multiclass Brier: mean(sum((p_HDA-y_HDA)^2)/3); lower is better.",
            "logLossNats": "Mean negative natural logarithm of the observed-outcome probability; lower is better.",
            "normalizedRps": "Mean two-threshold ranked probability score divided by two in H-D-A order; lower is better.",
            "topPickHitRateFractionalTies": "A tied top set receives 1/k credit when it contains the observed class and zero otherwise.",
            "topLabelEce10": "Ten equal-width confidence bins with fractional-tie correctness; weighted absolute calibration gap.",
            "macroClasswiseEce10": "Mean of ten-bin one-vs-rest ECE for H, D and A.",
        },
        "models": definitions,
        "leaderboard": leaderboard,
        "seasonSummaries": season_summaries,
        "calibration": calibration,
        "pairedUncertainty": {
            "referenceModelId": "market_closing_shin",
            "method": (
                "Deterministic paired percentile intervals from 5,000 ISO-week block-bootstrap resamples. "
                "Each sampled week retains both systems on identical fixtures. Model-minus-Shin differences below zero favour the model, "
                "but the closing reference has a later information set."
            ),
            "replicates": BOOTSTRAP_REPLICATES,
            "blockCount": len({match.block for match in evaluation}),
            "comparisons": paired_comparisons,
        },
        "rawVsCalibratedAblation": raw_calibrated_ablation,
        "pointInTimeBoundary": {
            "featureEventTimeSafe": True,
            "sameKickoffSafe": True,
            "sourceRevisionMode": "latest-state-publisher-csv",
            "sourceRevisionHistoryComplete": False,
            "correctionObservedAtComplete": False,
            "asOfReplayVerified": False,
            "productionActivationEligible": False,
            "explanation": (
                "The generator enforces event-time and declared 24-hour availability cutoffs, but the hash-pinned latest-state CSVs "
                "do not expose when later source corrections became observable. This research release therefore cannot activate a production model."
            ),
        },
        "source": {
            "publisher": "Football-Data.co.uk",
            "landingPage": "https://www.football-data.co.uk/data.php",
            "notes": "https://www.football-data.co.uk/notes.txt",
            "derivedFromManifest": {
                "repositoryPath": str(SOURCE_MANIFEST_PATH.relative_to(ROOT)),
                "sha256": source_manifest_sha,
            },
            "sourceInputs": source_inputs,
            "rawRowsRedistributed": False,
        },
        "reproducibility": {
            "generator": {
                "repositoryPath": str(GENERATOR_PATH.relative_to(ROOT)),
                "publicPath": PUBLIC_GENERATOR_PATH,
                "sha256": generator_sha,
            },
            "dependencyFiles": dependency_files,
            "dependencyVersions": {
                "python": platform.python_version(),
                **PACKAGE_VERSIONS,
            },
            "checkCommand": "python3 scripts/generate-epl-football-prediction-model-benchmark.py",
            "rebuildCommand": (
                "DYLD_LIBRARY_PATH=\"$PWD/.venv-model/libomp/libomp/22.1.8/lib\" "
                ".venv-model/bin/python scripts/generate-epl-football-prediction-model-benchmark.py --rebuild"
            ),
            "offlineRebuildCommand": (
                "DYLD_LIBRARY_PATH=\"$PWD/.venv-model/libomp/libomp/22.1.8/lib\" "
                ".venv-model/bin/python scripts/generate-epl-football-prediction-model-benchmark.py "
                "--rebuild --source-dir /path/to/hash-matched-sources"
            ),
            "writeCommand": (
                "DYLD_LIBRARY_PATH=\"$PWD/.venv-model/libomp/libomp/22.1.8/lib\" "
                ".venv-model/bin/python scripts/generate-epl-football-prediction-model-benchmark.py --write"
            ),
            "artifacts": {
                "json": PUBLIC_JSON_PATH,
                "csv": PUBLIC_CSV_PATH,
                "manifest": PUBLIC_MANIFEST_PATH,
            },
        },
        "rights": {
            "aggregateLicense": AGGREGATE_LICENSE,
            "aggregateRightsScope": AGGREGATE_RIGHTS_SCOPE,
            "sourceRightsStatus": "unknown/not asserted",
            "checkedAt": RELEASE_DATE,
            "rawSourceLicenseNotGranted": True,
            "rawRowsRedistributed": False,
        },
        "claimsBoundary": {
            "liveAccuracyClaim": False,
            "profitClaim": False,
            "bettingAdvice": False,
            "productionModelActivated": False,
            "algorithmCausalRankingClaim": False,
            "sameInformationMarketComparisonClaim": False,
            "fixtureLevelPredictionsPublished": False,
            "fittedArtifactsPublished": False,
        },
        "limitations": [
            "This aggregate benchmark covers one league and four evaluation seasons; it does not establish universal model superiority.",
            "Closing-market rows use later information than every 24-hour history-only model and are contextual references only.",
            "Complete-season fitting keeps one coherent leakage-safe comparison but does not update fitted coefficients or team effects within an evaluation season.",
            "Dynamic 14-feature and Elo snapshots may incorporate availability-safe earlier results during an evaluation season while their fitted classifier mapping remains frozen at the season boundary.",
            "The sequential Dixon-Coles row changes rho on shared ridge-Poisson rates and is not a full joint Dixon-Coles maximum-likelihood fit.",
            "Latest-state source CSVs lack correction observed-at history; hash pins freeze release bytes but cannot prove revision-time replay.",
            "Weekly block-bootstrap intervals preserve paired fixture weeks but do not prove independence, causality, profitability or a universal edge.",
            "No raw source rows, fixture probabilities, fitted estimators or calibrators are redistributed.",
        ],
    }
    json_bytes = canonical_json(artifact)
    csv_bytes = artifact_csv(artifact)
    manifest: dict[str, Any] = {
        "schemaVersion": "epl-football-prediction-model-benchmark-manifest/1.0.0",
        "benchmarkVersion": VERSION,
        "releaseDate": RELEASE_DATE,
        "aggregateLicense": AGGREGATE_LICENSE,
        "aggregateRightsScope": AGGREGATE_RIGHTS_SCOPE,
        "sourceRightsStatus": "unknown/not asserted",
        "rawSourceLicenseNotGranted": True,
        "rawRowsRedistributed": False,
        "publicManifestPath": PUBLIC_MANIFEST_PATH,
        "artifacts": [
            {
                "path": PUBLIC_JSON_PATH,
                "contentType": "application/json; charset=utf-8",
                "bytes": len(json_bytes),
                "sha256": sha256(json_bytes),
            },
            {
                "path": PUBLIC_CSV_PATH,
                "contentType": "text/csv; charset=utf-8",
                "bytes": len(csv_bytes),
                "sha256": sha256(csv_bytes),
            },
            {
                "path": PUBLIC_GENERATOR_PATH,
                "contentType": "text/x-python; charset=utf-8",
                "bytes": GENERATOR_PATH.stat().st_size,
                "sha256": generator_sha,
            },
        ],
        "sourceManifest": {
            "path": str(SOURCE_MANIFEST_PATH.relative_to(ROOT)),
            "sha256": source_manifest_sha,
        },
        "sourceInputs": source_inputs,
        "dependencyFiles": dependency_files,
        "dependencyVersions": {"python": platform.python_version(), **PACKAGE_VERSIONS},
        "generator": {
            "path": str(GENERATOR_PATH.relative_to(ROOT)),
            "publicPath": PUBLIC_GENERATOR_PATH,
            "sha256": generator_sha,
        },
    }
    return json_bytes, csv_bytes, canonical_json(manifest)


def verify_frozen() -> None:
    artifact_bytes = JSON_PATH.read_bytes()
    csv_bytes = CSV_PATH.read_bytes()
    manifest_bytes = MANIFEST_PATH.read_bytes()
    artifact = json.loads(artifact_bytes)
    manifest = json.loads(manifest_bytes)
    _, source_manifest_sha = source_contract()
    if artifact.get("schemaVersion") != VERSION:
        raise RuntimeError("Frozen benchmark schema version drifted.")
    evaluation = artifact.get("evaluation", {})
    if (
        evaluation.get("evaluationMatchCount") != EXPECTED_EVALUATION_MATCHES
        or evaluation.get("allModelsFullCoverage") is not True
        or len(artifact.get("leaderboard", [])) != len(MODEL_ORDER)
    ):
        raise RuntimeError("Frozen benchmark coverage contract drifted.")
    if artifact.get("pointInTimeBoundary", {}).get("productionActivationEligible") is not False:
        raise RuntimeError("Frozen production activation boundary is not fail-closed.")
    if manifest.get("sourceManifest", {}).get("sha256") != source_manifest_sha:
        raise RuntimeError("Frozen source-manifest hash drifted.")
    checks = {
        PUBLIC_JSON_PATH: artifact_bytes,
        PUBLIC_CSV_PATH: csv_bytes,
        PUBLIC_GENERATOR_PATH: GENERATOR_PATH.read_bytes(),
    }
    by_path = {entry["path"]: entry for entry in manifest.get("artifacts", [])}
    for path, value in checks.items():
        entry = by_path.get(path)
        if not entry or entry.get("bytes") != len(value) or entry.get("sha256") != sha256(value):
            raise RuntimeError(f"Frozen artifact hash mismatch: {path}.")
    if manifest.get("generator", {}).get("sha256") != sha256(GENERATOR_PATH.read_bytes()):
        raise RuntimeError("Frozen generator self-hash drifted.")
    for dependency in manifest.get("dependencyFiles", []):
        value = (ROOT / dependency["repositoryPath"]).read_bytes()
        if dependency.get("sha256") != sha256(value):
            raise RuntimeError(f"Frozen dependency hash mismatch: {dependency['repositoryPath']}.")
    print(
        "EPL same-match model benchmark frozen bytes verified: "
        + ", ".join(f"{path}={sha256(value)}" for path, value in checks.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify committed bytes and hashes (default)")
    parser.add_argument("--rebuild", action="store_true", help="rerun hash-pinned benchmark and compare")
    parser.add_argument("--write", action="store_true", help="rerun benchmark and write immutable v1 artifacts")
    parser.add_argument("--source-dir", type=Path, help="directory containing user-held <season>/E0.csv sources")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.check, args.rebuild, args.write)) > 1:
        parser.error("choose only one of --check, --rebuild or --write")
    if args.source_dir and not (args.rebuild or args.write):
        parser.error("--source-dir requires --rebuild or --write")
    if not args.rebuild and not args.write:
        verify_frozen()
        return
    built = build_release(args.source_dir.resolve() if args.source_dir else None)
    if args.write:
        for path, value in zip((JSON_PATH, CSV_PATH, MANIFEST_PATH), built, strict=True):
            path.write_bytes(value)
            print(f"{path.relative_to(ROOT)} {sha256(value)}")
        return
    current = (JSON_PATH.read_bytes(), CSV_PATH.read_bytes(), MANIFEST_PATH.read_bytes())
    if current != built:
        raise RuntimeError("Full same-match benchmark rebuild drifted from immutable v1.")
    verify_frozen()


if __name__ == "__main__":
    main()
