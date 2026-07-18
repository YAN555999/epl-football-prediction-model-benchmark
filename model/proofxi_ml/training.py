from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import warnings
import zipfile
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, TrainingExample
from .metrics import evaluate_probabilities


ALGORITHM = "LightGBM multiclass + one-vs-rest isotonic"
POINT_IN_TIME_PROVENANCE_SCHEMA_VERSION = "proofxi-point-in-time-provenance-v1"
POINT_IN_TIME_REVISION_MODE = "versioned-observed-at-history"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    folds: int = 3
    minimum_raw_train: int = 300
    minimum_calibration: int = 100
    random_seed: int = 20260710
    n_estimators: int = 350
    learning_rate: float = 0.035
    num_leaves: int = 24
    min_child_samples: int = 35
    subsample: float = 0.85
    colsample_bytree: float = 0.85


class ConstantCalibrator:
    """Pickle-safe fallback when a calibration slice has one binary class."""

    def __init__(self, value: float) -> None:
        self.value = min(1.0, max(0.0, float(value)))

    def predict(self, values: Sequence[float]) -> list[float]:
        return [self.value for _ in values]


def _heavy_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import joblib
        import numpy as np
        from lightgbm import LGBMClassifier
        from sklearn.isotonic import IsotonicRegression
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Training requires model/requirements-ml.txt and the platform OpenMP runtime "
            "(libomp on macOS); install them in an isolated environment."
        ) from exc
    return joblib, np, LGBMClassifier, IsotonicRegression


def _matrix(examples: Sequence[TrainingExample], np: Any) -> Any:
    return np.asarray(
        [
            [
                np.nan if example.snapshot.values[name] is None else example.snapshot.values[name]
                for name in FEATURE_NAMES
            ]
            for example in examples
        ],
        dtype=float,
    )


def _targets(examples: Sequence[TrainingExample], np: Any) -> Any:
    return np.asarray([example.target for example in examples], dtype=int)


def _model(config: TrainingConfig, classifier: Any) -> Any:
    return classifier(
        objective="multiclass",
        num_class=3,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=0.15,
        reg_lambda=0.4,
        random_state=config.random_seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _fit_calibrators(raw: Any, targets: Any, isotonic: Any, np: Any) -> list[Any]:
    calibrators: list[Any] = []
    for class_index in range(3):
        binary = (targets == class_index).astype(float)
        if len(np.unique(binary)) < 2 or len(np.unique(raw[:, class_index])) < 2:
            calibrators.append(ConstantCalibrator(float(np.mean(binary))))
            continue
        calibrator = isotonic(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw[:, class_index], binary)
        calibrators.append(calibrator)
    return calibrators


def calibrate_probabilities(raw: Any, calibrators: Sequence[Any], np: Any) -> Any:
    calibrated = np.column_stack(
        [
            np.asarray(calibrator.predict(raw[:, class_index]), dtype=float)
            for class_index, calibrator in enumerate(calibrators)
        ]
    )
    calibrated = np.clip(calibrated, 1e-9, 1.0)
    row_sums = calibrated.sum(axis=1, keepdims=True)
    invalid = ~np.isfinite(row_sums) | (row_sums <= 0)
    if invalid.any():
        calibrated[invalid[:, 0], :] = 1.0 / 3.0
        row_sums = calibrated.sum(axis=1, keepdims=True)
    return calibrated / row_sums


def predict_raw_probabilities(model: Any, matrix: Any) -> Any:
    # LightGBM 4.6 + sklearn 1.7 emits this warning for ndarray input even
    # though the model was fitted with the same locked ndarray column order.
    # Bundle metadata independently verifies that exact order before inference.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        return model.predict_proba(matrix, validate_features=False)


def _kickoff_group_boundaries(
    ordered: Sequence[TrainingExample],
) -> list[int]:
    """Return row offsets that never split one kick-off timestamp."""

    boundaries = [0]
    for index in range(1, len(ordered)):
        if ordered[index - 1].fixture.kickoff_utc != ordered[index].fixture.kickoff_utc:
            boundaries.append(index)
    if ordered:
        boundaries.append(len(ordered))
    return boundaries


@dataclass(frozen=True, slots=True)
class FoldWindow:
    raw_end: int
    calibration_start: int
    calibration_end: int
    evaluation_start: int
    evaluation_end: int

    @property
    def raw_calibration_purged_count(self) -> int:
        return self.calibration_start - self.raw_end

    @property
    def calibration_evaluation_purged_count(self) -> int:
        return self.evaluation_start - self.calibration_end


def _prefix_target_availability(
    ordered: Sequence[TrainingExample],
) -> list[int]:
    prefix = [-1]
    for example in ordered:
        prefix.append(max(prefix[-1], example.target_available_utc))
    return prefix


def _latest_safe_group_boundary(
    group_boundaries: Sequence[int],
    boundary_target_availability: Sequence[int],
    *,
    upper_row: int,
    strictly_before_utc: int,
) -> int:
    row_index = bisect_right(group_boundaries, upper_row) - 1
    time_index = bisect_left(boundary_target_availability, strictly_before_utc) - 1
    selected_index = min(row_index, time_index)
    if selected_index < 0:
        raise ValueError("no availability-safe kickoff boundary exists")
    return group_boundaries[selected_index]


def _stage_window_before_evaluation(
    ordered: Sequence[TrainingExample],
    group_boundaries: Sequence[int],
    boundary_target_availability: Sequence[int],
    *,
    evaluation_start: int,
    minimum_raw_train: int,
    minimum_calibration: int,
) -> tuple[int, int, int]:
    if evaluation_start >= len(ordered):
        raise ValueError("evaluation_start must point to an evaluation row")
    evaluation_as_of = ordered[evaluation_start].snapshot.as_of_utc

    # Work backwards from the evaluation boundary. Rows between calibration_end
    # and evaluation_start are explicitly purged until every calibration label
    # was available strictly before the first evaluation prediction timestamp.
    calibration_end = _latest_safe_group_boundary(
        group_boundaries,
        boundary_target_availability,
        upper_row=evaluation_start,
        strictly_before_utc=evaluation_as_of,
    )
    calibration_start_index = (
        bisect_right(group_boundaries, calibration_end - minimum_calibration) - 1
    )
    if calibration_start_index < 0:
        raise ValueError(
            "not enough availability-safe kickoff groups for a calibration window"
        )
    calibration_start = group_boundaries[calibration_start_index]
    if calibration_start >= calibration_end:
        raise ValueError("availability-safe calibration window cannot be empty")
    calibration_as_of = ordered[calibration_start].snapshot.as_of_utc
    raw_end = _latest_safe_group_boundary(
        group_boundaries,
        boundary_target_availability,
        upper_row=calibration_start,
        strictly_before_utc=calibration_as_of,
    )
    if raw_end < minimum_raw_train:
        raise ValueError(
            "not enough availability-safe kickoff groups for the minimum raw-training window"
        )
    return raw_end, calibration_start, calibration_end


def _final_fit_boundaries(
    ordered: Sequence[TrainingExample],
    *,
    minimum_raw_train: int,
    minimum_calibration: int,
) -> tuple[int, int]:
    group_boundaries = _kickoff_group_boundaries(ordered)
    prefix_target_availability = _prefix_target_availability(ordered)
    boundary_target_availability = [
        prefix_target_availability[boundary] for boundary in group_boundaries
    ]
    total = len(ordered)
    calibration_start_index = (
        bisect_right(group_boundaries, total - minimum_calibration) - 1
    )
    if calibration_start_index < 0:
        raise ValueError("not enough kickoff groups for final calibration")
    calibration_start = group_boundaries[calibration_start_index]
    if calibration_start >= total:
        raise ValueError("final calibration window cannot be empty")
    calibration_as_of = ordered[calibration_start].snapshot.as_of_utc
    raw_end = _latest_safe_group_boundary(
        group_boundaries,
        boundary_target_availability,
        upper_row=calibration_start,
        strictly_before_utc=calibration_as_of,
    )
    if raw_end < minimum_raw_train:
        raise ValueError(
            "not enough availability-safe kickoff groups for final raw training and calibration"
        )
    return raw_end, calibration_start


def _fold_boundaries(
    ordered: Sequence[TrainingExample],
    config: TrainingConfig,
) -> tuple[int, int, list[FoldWindow]]:
    if config.folds < 3:
        raise ValueError("walk-forward backtesting requires at least three folds")
    total = len(ordered)
    minimum_raw_train = max(config.minimum_raw_train, int(total * 0.35))
    minimum_calibration = max(config.minimum_calibration, int(total * 0.10))
    group_boundaries = _kickoff_group_boundaries(ordered)
    prefix_target_availability = _prefix_target_availability(ordered)
    boundary_target_availability = [
        prefix_target_availability[boundary] for boundary in group_boundaries
    ]
    if any(
        left.snapshot.as_of_utc > right.snapshot.as_of_utc
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("training prediction timestamps must be chronological")
    group_count = max(0, len(group_boundaries) - 1)
    if group_count < config.folds + 2:
        raise ValueError(
            "not enough kickoff groups for raw training, calibration and "
            f"{config.folds} walk-forward folds"
        )

    first_evaluation_boundary_index: int | None = None
    first_stage_window: tuple[int, int, int] | None = None
    for boundary_index in range(2, len(group_boundaries)):
        evaluation_start = group_boundaries[boundary_index]
        remaining_groups = group_count - boundary_index
        if remaining_groups < config.folds:
            break
        try:
            stage_window = _stage_window_before_evaluation(
                ordered,
                group_boundaries[: boundary_index + 1],
                boundary_target_availability[: boundary_index + 1],
                evaluation_start=evaluation_start,
                minimum_raw_train=minimum_raw_train,
                minimum_calibration=minimum_calibration,
            )
        except ValueError:
            continue
        first_evaluation_boundary_index = boundary_index
        first_stage_window = stage_window
        break

    if first_evaluation_boundary_index is None or first_stage_window is None:
        raise ValueError(
            "not enough complete kickoff groups for "
            f"{config.folds} walk-forward folds after training and calibration"
        )

    evaluation_start = group_boundaries[first_evaluation_boundary_index]
    evaluation_rows = total - evaluation_start
    fold_end_boundary_indices: list[int] = []
    previous_boundary_index = first_evaluation_boundary_index
    for fold_index in range(1, config.folds):
        target = evaluation_start + evaluation_rows * fold_index / config.folds
        minimum_index = previous_boundary_index + 1
        maximum_index = group_count - (config.folds - fold_index)
        if minimum_index > maximum_index:
            raise ValueError(
                "walk-forward folds cannot be formed without splitting a kickoff group"
            )
        selected = min(
            range(minimum_index, maximum_index + 1),
            key=lambda index: (abs(group_boundaries[index] - target), index),
        )
        fold_end_boundary_indices.append(selected)
        previous_boundary_index = selected
    fold_end_boundary_indices.append(group_count)

    windows: list[FoldWindow] = []
    current_evaluation_boundary_index = first_evaluation_boundary_index
    for evaluation_end_boundary_index in fold_end_boundary_indices:
        current_evaluation_start = group_boundaries[current_evaluation_boundary_index]
        raw_end, calibration_start, calibration_end = _stage_window_before_evaluation(
            ordered,
            group_boundaries[: current_evaluation_boundary_index + 1],
            boundary_target_availability[: current_evaluation_boundary_index + 1],
            evaluation_start=current_evaluation_start,
            minimum_raw_train=minimum_raw_train,
            minimum_calibration=minimum_calibration,
        )
        evaluation_end = group_boundaries[evaluation_end_boundary_index]
        if evaluation_end <= current_evaluation_start:
            raise ValueError("walk-forward evaluation fold cannot be empty")
        windows.append(
            FoldWindow(
                raw_end=raw_end,
                calibration_start=calibration_start,
                calibration_end=calibration_end,
                evaluation_start=current_evaluation_start,
                evaluation_end=evaluation_end,
            )
        )
        current_evaluation_boundary_index = evaluation_end_boundary_index

    return first_stage_window[0], minimum_calibration, windows


def walk_forward_backtest(
    examples: Iterable[TrainingExample],
    config: TrainingConfig = TrainingConfig(),
) -> tuple[dict[str, Any], Any, list[Any]]:
    joblib, np, classifier, isotonic = _heavy_dependencies()
    del joblib
    ordered = sorted(
        examples,
        key=lambda item: (item.fixture.kickoff_utc, item.fixture.fixture_id),
    )
    if not ordered:
        raise ValueError("no final fixtures are available for training")
    initial_train, minimum_calibration, boundaries = _fold_boundaries(ordered, config)
    all_targets: list[int] = []
    all_probabilities: list[list[float]] = []
    fold_reports: list[dict[str, Any]] = []

    for fold_index, window in enumerate(boundaries, start=1):
        raw_train = ordered[: window.raw_end]
        calibration = ordered[window.calibration_start : window.calibration_end]
        evaluation = ordered[window.evaluation_start : window.evaluation_end]
        x_train = _matrix(raw_train, np)
        y_train = _targets(raw_train, np)
        if len(np.unique(y_train)) != 3:
            raise ValueError(f"fold {fold_index} raw training data does not contain all outcomes")
        raw_model = _model(config, classifier)
        raw_model.fit(x_train, y_train)
        raw_calibration = predict_raw_probabilities(raw_model, _matrix(calibration, np))
        calibrators = _fit_calibrators(
            raw_calibration,
            _targets(calibration, np),
            isotonic,
            np,
        )
        raw_evaluation = predict_raw_probabilities(raw_model, _matrix(evaluation, np))
        probabilities = calibrate_probabilities(raw_evaluation, calibrators, np)
        targets = [item.target for item in evaluation]
        fold_metrics = evaluate_probabilities(targets, probabilities.tolist())
        all_targets.extend(targets)
        all_probabilities.extend(probabilities.tolist())
        fold_reports.append(
            {
                "fold": fold_index,
                "rawTrainCount": len(raw_train),
                "calibrationCount": len(calibration),
                "evaluationCount": len(evaluation),
                "rawCalibrationPurgedCount": window.raw_calibration_purged_count,
                "calibrationEvaluationPurgedCount": (
                    window.calibration_evaluation_purged_count
                ),
                "purgedCount": (
                    window.raw_calibration_purged_count
                    + window.calibration_evaluation_purged_count
                ),
                "evaluationStartUtc": evaluation[0].fixture.kickoff_utc,
                "evaluationEndUtc": evaluation[-1].fixture.kickoff_utc,
                **fold_metrics,
            }
        )

    aggregate = evaluate_probabilities(all_targets, all_probabilities)
    evaluation_start_utc = fold_reports[0]["evaluationStartUtc"]
    evaluation_end_utc = fold_reports[-1]["evaluationEndUtc"]
    report: dict[str, Any] = {
        "protocol": "walk-forward",
        "sampleCount": len(all_targets),
        "foldCount": len(fold_reports),
        "evaluationStartUtc": evaluation_start_utc,
        "evaluationEndUtc": evaluation_end_utc,
        "evaluationCoverageDays": round(
            (evaluation_end_utc - evaluation_start_utc) / 86_400,
            3,
        ),
        **aggregate,
        "folds": fold_reports,
        "gate": {
            "required": {
                "hitRate": ">0.48",
                "brier": "<0.22",
                "sampleCount": ">=500",
                "foldCount": ">=3",
                "evaluationCoverageDays": ">=365",
                "pointInTimeProvenance": (
                    "complete versioned observed-at revision history covering the "
                    "evaluation window"
                ),
            }
        },
    }
    apply_candidate_gate(report)

    # The production artifact uses the same leakage-safe split: the last
    # chronological slice calibrates a model fitted only on earlier rows.
    final_raw_end, final_calibration_start = _final_fit_boundaries(
        ordered,
        minimum_raw_train=initial_train,
        minimum_calibration=minimum_calibration,
    )
    final_raw_train = ordered[:final_raw_end]
    final_calibration = ordered[final_calibration_start:]
    final_model = _model(config, classifier)
    final_model.fit(_matrix(final_raw_train, np), _targets(final_raw_train, np))
    final_raw_calibration = predict_raw_probabilities(
        final_model,
        _matrix(final_calibration, np),
    )
    final_calibrators = _fit_calibrators(
        final_raw_calibration,
        _targets(final_calibration, np),
        isotonic,
        np,
    )
    report["finalFit"] = {
        "rawTrainCount": len(final_raw_train),
        "calibrationCount": len(final_calibration),
        "rawCalibrationPurgedCount": final_calibration_start - final_raw_end,
        "purgedCount": final_calibration_start - final_raw_end,
        "trainingCutoffUtc": ordered[-1].fixture.kickoff_utc,
    }
    return report, final_model, final_calibrators


def gate_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not (float(report.get("hitRate", -1)) > 0.48):
        failures.append("hitRate must be > 0.48")
    if not (float(report.get("brier", math.inf)) < 0.22):
        failures.append("brier must be < 0.22")
    if int(report.get("sampleCount", 0)) < 500:
        failures.append("sampleCount must be >= 500")
    if int(report.get("foldCount", 0)) < 3:
        failures.append("foldCount must be >= 3")
    start = int(report.get("evaluationStartUtc", 0))
    end = int(report.get("evaluationEndUtc", 0))
    if end - start < 365 * 86_400:
        failures.append("evaluation coverage must be >= 365 days")
    if str(report.get("protocol")) != "walk-forward":
        failures.append("protocol must be walk-forward")
    return failures


def point_in_time_provenance_failures(report: dict[str, Any]) -> list[str]:
    """Validate the evidence needed to replay every training row as it was known.

    A latest-state provider snapshot is insufficient because a later score or
    status correction can otherwise leak into an earlier historical feature
    snapshot. The manifest digest binds the completeness claim to one immutable
    revision inventory; activation independently rechecks the same contract.
    """

    proof = report.get("pointInTimeProvenance")
    if not isinstance(proof, dict):
        return ["pointInTimeProvenance is required"]

    failures: list[str] = []
    if proof.get("schemaVersion") != POINT_IN_TIME_PROVENANCE_SCHEMA_VERSION:
        failures.append(
            "pointInTimeProvenance.schemaVersion must be "
            + POINT_IN_TIME_PROVENANCE_SCHEMA_VERSION
        )
    if proof.get("sourceRevisionMode") != POINT_IN_TIME_REVISION_MODE:
        failures.append(
            "pointInTimeProvenance.sourceRevisionMode must be "
            + POINT_IN_TIME_REVISION_MODE
        )
    for field in (
        "sourceRevisionHistoryComplete",
        "correctionObservedAtComplete",
        "asOfReplayVerified",
    ):
        if proof.get(field) is not True:
            failures.append(f"pointInTimeProvenance.{field} must be true")

    manifest_sha256 = proof.get("revisionManifestSha256")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in manifest_sha256
    ):
        failures.append(
            "pointInTimeProvenance.revisionManifestSha256 must be a SHA-256 digest"
        )

    def integer_field(name: str) -> int | None:
        value = proof.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            failures.append(f"pointInTimeProvenance.{name} must be a non-negative integer")
            return None
        return value

    fixture_count = integer_field("fixtureCount")
    revision_count = integer_field("revisionCount")
    coverage_start = integer_field("coverageStartUtc")
    coverage_end = integer_field("coverageEndUtc")
    sample_count = report.get("sampleCount")
    evaluation_start = report.get("evaluationStartUtc")
    evaluation_end = report.get("evaluationEndUtc")
    if (
        fixture_count is not None
        and isinstance(sample_count, int)
        and fixture_count < sample_count
    ):
        failures.append(
            "pointInTimeProvenance.fixtureCount must cover every evaluation sample"
        )
    if (
        fixture_count is not None
        and revision_count is not None
        and revision_count < fixture_count
    ):
        failures.append(
            "pointInTimeProvenance.revisionCount cannot be less than fixtureCount"
        )
    if (
        coverage_start is not None
        and isinstance(evaluation_start, int)
        and coverage_start > evaluation_start
    ):
        failures.append(
            "pointInTimeProvenance coverage must start by evaluationStartUtc"
        )
    if (
        coverage_end is not None
        and isinstance(evaluation_end, int)
        and coverage_end < evaluation_end
    ):
        failures.append(
            "pointInTimeProvenance coverage must extend through evaluationEndUtc"
        )
    if (
        coverage_start is not None
        and coverage_end is not None
        and coverage_end < coverage_start
    ):
        failures.append(
            "pointInTimeProvenance coverageEndUtc cannot precede coverageStartUtc"
        )
    return failures


def candidate_gate_failures(report: dict[str, Any]) -> list[str]:
    return gate_failures(report) + point_in_time_provenance_failures(report)


def apply_candidate_gate(report: dict[str, Any]) -> list[str]:
    failures = candidate_gate_failures(report)
    gate = report.setdefault("gate", {})
    if not isinstance(gate, dict):
        gate = {}
        report["gate"] = gate
    gate["passed"] = not failures
    gate["failures"] = failures
    return failures


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    configured = os.environ.get("GITHUB_SHA", "").strip()
    if configured and all(character in "0123456789abcdefABCDEF" for character in configured):
        return configured.lower()
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if value and all(character in "0123456789abcdefABCDEF" for character in value):
            return value.lower()
    except (OSError, subprocess.CalledProcessError):
        pass
    raise RuntimeError("a real code Git SHA is required to create a candidate")


def write_backtest_report(output_dir: str | Path, report: dict[str, Any]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "backtest-report.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def create_candidate(
    output_dir: str | Path,
    report: dict[str, Any],
    model: Any,
    calibrators: Sequence[Any],
    *,
    training_cutoff_utc: int,
    config: TrainingConfig,
    version: str | None = None,
) -> dict[str, Any] | None:
    if candidate_gate_failures(report):
        return None
    joblib, _, _, _ = _heavy_dependencies()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_version = version or f"fpai-lgbm-v1-{timestamp}"
    root = Path(output_dir)
    candidate_dir = root / "candidates" / model_version
    candidate_dir.mkdir(parents=True, exist_ok=False)
    model_path = candidate_dir / "model.joblib"
    calibrator_path = candidate_dir / "calibrators.joblib"
    joblib.dump(model, model_path, compress=3)
    joblib.dump(list(calibrators), calibrator_path, compress=3)
    model_sha = _sha256_file(model_path)
    calibrator_sha = _sha256_file(calibrator_path)
    code_sha = _git_sha()
    register_payload = {
        "version": model_version,
        "algorithm": ALGORITHM,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "featureNames": list(FEATURE_NAMES),
        "trainingCutoffUtc": training_cutoff_utc,
        "codeGitSha": code_sha,
        "modelBundleSha256": model_sha,
        "calibratorSha256": calibrator_sha,
        "backtestMetrics": report,
    }
    metadata = {
        "bundleSchemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "version": model_version,
        "algorithm": ALGORITHM,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "featureNames": list(FEATURE_NAMES),
        "trainingCutoffUtc": training_cutoff_utc,
        "codeGitSha": code_sha,
        "modelSha256": model_sha,
        "calibratorSha256": calibrator_sha,
        "trainingConfig": asdict(config),
    }
    for name, value in (
        ("register.json", register_payload),
        ("metadata.json", metadata),
        ("backtest-report.json", report),
    ):
        (candidate_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    archive = root / f"{model_version}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in (
            "model.joblib",
            "calibrators.joblib",
            "register.json",
            "metadata.json",
            "backtest-report.json",
        ):
            bundle.write(candidate_dir / name, arcname=name)
    pointer = {
        "version": model_version,
        "candidateDir": str(candidate_dir.resolve()),
        "archive": str(archive.resolve()),
        "archiveSha256": _sha256_file(archive),
        "registerPayload": str((candidate_dir / "register.json").resolve()),
    }
    (root / "latest-candidate.json").write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pointer
