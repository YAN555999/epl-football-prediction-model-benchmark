# Reproducibility contract and pre-release audit trail

This archive separates two claims:

1. **Frozen-byte integrity:** the committed JSON, CSV, manifest, generator,
   source manifest and model dependency hashes match the immutable v1 contract.
2. **Full regeneration:** the locked generator downloads or reads the ten
   SHA-pinned source files, refits every system, rebuilds all aggregates and
   compares the three generated outputs byte-for-byte with v1.

The first claim is fast, platform-independent and network-free. The second
claim depends on a declared Python, package, CPU and math-runtime environment.
The repository does not claim that floating-point model fitting is bitwise
portable across every possible platform.

## Locked software

- Python 3.13.2
- joblib 1.5.2
- LightGBM 4.6.0
- NumPy 2.2.6
- scikit-learn 1.7.2
- SciPy 1.18.0

The exact Python package requirements are committed in `requirements.txt`.
The generator additionally binds the exact `features.py` and `training.py`
bytes in the release manifest.

## Pre-release evidence

Before tag `v1.0.0` was created:

- A macOS arm64 rebuild using Python 3.13.2 and the locked package versions
  reproduced all three outputs exactly.
- The first Ubuntu 24.04 x64 CI rebuild attempt reported a byte comparison
  failure. It did not write or publish replacement outputs, and the failed
  attempt remains visible in the public Actions history.
- A dedicated aggregate-only diagnostic run then used `--write` on Ubuntu
  24.04 x64. The rebuilt JSON, CSV and manifest matched all three frozen
  SHA-256 values exactly, with zero changed numeric fields or CSV cells.
- Two subsequent reruns of the original fail-closed `--rebuild` job reproduced
  all three files exactly.

Because the first failure was not silently discarded or explainable from a
source change, the release CI was strengthened before tagging. Its regeneration
job now performs two consecutive full rebuilds in the same locked Linux x64
job, and both must be byte-identical. The separate fast frozen-byte job remains
mandatory.

This audit history is evidence of the tested contract, not proof that every
future runner or hardware target must produce bitwise-identical floating-point
results. A mismatch fails CI and requires diagnosis; it is never normalized
away, rounded away after the fact or converted into a passing result.
