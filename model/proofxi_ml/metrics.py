from __future__ import annotations

import math
from typing import Iterable, Sequence


def _validated(
    targets: Iterable[int], probabilities: Iterable[Sequence[float]]
) -> tuple[list[int], list[tuple[float, float, float]]]:
    y_true = [int(value) for value in targets]
    rows: list[tuple[float, float, float]] = []
    for index, values in enumerate(probabilities):
        if len(values) != 3:
            raise ValueError(f"probability row {index} must contain three values")
        row = tuple(float(value) for value in values)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in row):
            raise ValueError(f"probability row {index} contains an invalid value")
        if abs(sum(row) - 1.0) > 1e-6:
            raise ValueError(f"probability row {index} does not sum to one")
        rows.append(row)  # type: ignore[arg-type]
    if len(y_true) != len(rows) or not y_true:
        raise ValueError("targets and probabilities must have equal non-zero lengths")
    if any(value not in {0, 1, 2} for value in y_true):
        raise ValueError("targets must be 0, 1 or 2")
    return y_true, rows


def multiclass_brier(
    targets: Iterable[int], probabilities: Iterable[Sequence[float]]
) -> float:
    """Return the class-averaged Brier score used by the public model ledger."""
    y_true, rows = _validated(targets, probabilities)
    total = 0.0
    for target, row in zip(y_true, rows):
        total += sum(
            (probability - (1.0 if class_index == target else 0.0)) ** 2
            for class_index, probability in enumerate(row)
        ) / 3.0
    return total / len(rows)


def multiclass_log_loss(
    targets: Iterable[int], probabilities: Iterable[Sequence[float]]
) -> float:
    y_true, rows = _validated(targets, probabilities)
    epsilon = 1e-15
    return -sum(math.log(max(epsilon, rows[index][target])) for index, target in enumerate(y_true)) / len(rows)


def confidence_buckets(
    targets: Iterable[int], probabilities: Iterable[Sequence[float]]
) -> list[dict[str, float | int | str | None]]:
    y_true, rows = _validated(targets, probabilities)
    edges = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001)
    output: list[dict[str, float | int | str | None]] = []
    for lower, upper in zip(edges, edges[1:]):
        selected: list[tuple[float, bool]] = []
        for target, row in zip(y_true, rows):
            pick = max(range(3), key=lambda index: row[index])
            confidence = row[pick]
            if lower <= confidence < upper:
                selected.append((confidence, pick == target))
        label_upper = 1.0 if upper > 1 else upper
        output.append(
            {
                "range": f"{lower:.1f}-{label_upper:.1f}",
                "lower": lower,
                "upper": label_upper,
                "count": len(selected),
                "meanConfidence": (
                    sum(item[0] for item in selected) / len(selected) if selected else None
                ),
                "observedHitRate": (
                    sum(1 for item in selected if item[1]) / len(selected)
                    if selected
                    else None
                ),
            }
        )
    return output


def expected_calibration_error(
    targets: Iterable[int],
    probabilities: Iterable[Sequence[float]],
    *,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least two")
    y_true, rows = _validated(targets, probabilities)
    total = len(rows)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected: list[tuple[float, bool]] = []
        for target, row in zip(y_true, rows):
            pick = max(range(3), key=lambda index: row[index])
            confidence = row[pick]
            if lower <= confidence < upper or (
                bin_index == bins - 1 and confidence == 1.0
            ):
                selected.append((confidence, pick == target))
        if not selected:
            continue
        mean_confidence = sum(item[0] for item in selected) / len(selected)
        accuracy = sum(1 for item in selected if item[1]) / len(selected)
        error += len(selected) / total * abs(accuracy - mean_confidence)
    return error


def evaluate_probabilities(
    targets: Iterable[int], probabilities: Iterable[Sequence[float]]
) -> dict[str, object]:
    y_true, rows = _validated(targets, probabilities)
    hits = 0
    for target, row in zip(y_true, rows):
        if max(range(3), key=lambda index: row[index]) == target:
            hits += 1
    return {
        "hitRate": hits / len(rows),
        "brier": multiclass_brier(y_true, rows),
        "logLoss": multiclass_log_loss(y_true, rows),
        "ece": expected_calibration_error(y_true, rows),
        "confidenceBuckets": confidence_buckets(y_true, rows),
    }
