# EPL football prediction model benchmark: Elo, LightGBM, Poisson and Dixon-Coles

[![Verify immutable benchmark](https://github.com/YAN555999/epl-football-prediction-model-benchmark/actions/workflows/verify.yml/badge.svg)](https://github.com/YAN555999/epl-football-prediction-model-benchmark/actions/workflows/verify.yml)

**Canonical publication:** <https://footballproofai.com/research/epl-football-prediction-model-benchmark>

This is the public, version-pinned research archive for Football Proof AI's
same-match English Premier League 1X2 model benchmark. Ten systems were scored
on the identical 1,520 completed fixtures across four full evaluation seasons,
with complete earlier seasons used for expanding-origin training.

- Immutable release: <https://github.com/YAN555999/epl-football-prediction-model-benchmark/releases/tag/v1.0.0>
- Evaluation: 1,520 EPL matches, 2022/23 through 2025/26
- Prior history: six complete EPL seasons, 2016/17 through 2021/22
- Models: uniform and expanding-prior baselines; Elo-logit; 14-feature logistic regression; ridge-Poisson; sequential Dixon-Coles; raw and isotonic-calibrated LightGBM
- Later-information references: proportional and Shin de-vig closing odds
- Publication boundary: aggregate metrics only; no fixture rows, team names, scorelines, odds triplets, fitted estimators or fixture-level probabilities are redistributed

## Direct result

On the exact same 1,520 fixtures, Elo to multinomial logistic regression had
the lowest history-only class-averaged Brier score: **0.1951354035**. The Shin
closing-market reference scored **0.1900581029**, but it contains later
information and is not a like-for-like 24-hour-ahead comparator.

| System | Information set | Brier (lower is better) | Log loss | Top-pick hit rate |
| --- | --- | ---: | ---: | ---: |
| Uniform 1/3 | Fixed distribution | 0.222222 | 1.098612 | 33.33% fractional ties |
| Expanding EPL prior | Past results, 24h safe | 0.215258 | 1.067752 | 44.47% |
| Elo to multinomial logit | History features, 24h safe | **0.195135** | **0.981618** | **53.62%** |
| 14-feature multinomial logistic | History features, 24h safe | 0.196004 | 0.984332 | 53.16% |
| Independent ridge-Poisson | Past results, 24h safe | 0.205149 | 1.027320 | 49.41% |
| Sequential Dixon-Coles | Past results, 24h safe | 0.205142 | 1.027709 | 49.41% |
| Raw 14-feature LightGBM | History features, 24h safe | 0.212892 | 1.075930 | 47.83% |
| LightGBM + training-only isotonic | History features, 24h safe | 0.203274 | 1.030579 | 50.20% |
| Proportional de-vig closing market | Closing market, later information | 0.190072 | 0.960310 | 55.13% |
| Shin de-vig closing market | Closing market, later information | 0.190058 | 0.960009 | 55.13% |

Training-only isotonic calibration improved raw LightGBM's Brier score by
0.0096186332. The paired 95% weekly-block bootstrap interval for calibrated
minus raw was [-0.0133573967, -0.0059268493]. This is evidence for calibration
in this locked benchmark, not a universal ranking of algorithms.

## What “same-match” means

Every system covers all 1,520 evaluation fixtures and carries the same fixture
key-set digest:
`be42108329815e8969c95eace1537d855bb8bcbce148800254a01b19836c7151`.
Evaluation folds are complete seasons. For each fold, only complete earlier
seasons are used for fitting. Historical results become feature-eligible after
a declared 24-hour availability lag; feature snapshots are made 24 hours
before kick-off. LightGBM isotonic calibrators use only expanding earlier-season
out-of-fold probabilities.

## Immutable v1 artifacts

| Repository file | SHA-256 |
| --- | --- |
| `data/epl-football-prediction-model-benchmark-1.0.0.json` | `60eedfeca8f0e8b0bad76f65734fbd89797f5246910aea4a2e86553857d1cde1` |
| `data/epl-football-prediction-model-benchmark-1.0.0.csv` | `ecdbdf708c649791de3f118798c110e0ca9c2cbc5c026760e2634d5d4e0e746a` |
| `data/epl-football-prediction-model-benchmark-1.0.0-manifest.json` | `8037b509f57984b5c076ace79917a5c6868a3136e82d6737fbdbb1a6635a0e9b` |
| `scripts/generate-epl-football-prediction-model-benchmark.py` | `ae61549bf96803665660b153a20a973d29faa18d17e982e344a9c99ff9dd06a3` |
| `data/football-1x2-empirical-benchmark-1.0.0-manifest.json` | `801f8c6c23e30f945dcba38e4c256fd4e9f3652f7e2ea965e4788f0e7cf9014a` |

Release attachments contain these exact bytes plus `SHA256SUMS`. The benchmark
manifest binds the outputs to the ten declared source URLs and SHA-256 values,
the exact generator, and the two model dependency files used in the release.

## Verify quickly

The fast contract needs only Python's standard library and does not access the
network:

```bash
python scripts/generate-epl-football-prediction-model-benchmark.py --check
python -m unittest discover -s tests -p 'test_*.py'
```

## Rebuild byte-for-byte

The locked environment is Python 3.13.2 with exact package versions in
`requirements.txt`:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/generate-epl-football-prediction-model-benchmark.py --rebuild
```

The rebuild downloads the ten declared source files from their publisher into
memory, verifies each source SHA-256 before parsing, rebuilds the benchmark and
compares all three generated outputs byte-for-byte with v1. No downloaded raw
row is cached or committed. Alternatively, `--source-dir` accepts user-held
`<season>/E0.csv` inputs after applying the same source hash checks.

If an upstream correction changes a byte, the source pin fails closed. Version
1.0.0 is never rewritten; a reviewed correction requires a new version, new
hashes and a new release.

## Honest interpretation boundary

- The latest-state publisher CSVs do not expose when historical corrections became observable. Event-time and 24-hour availability rules are enforced, but full correction-observed-at replay is not proven.
- Therefore this research artifact is explicitly **not eligible to activate a production prediction model**.
- The closing-market rows contain later information and cannot support a same-information algorithm-versus-market claim.
- Brier score, log loss, RPS, calibration error and hit rate do not prove profit.
- This archive is first-party reproducible research, not independent peer review and not a third-party publication timestamp.
- There are no live-accuracy, profit, betting-advice or causal model-ranking claims in v1.

## Citation

Use [`CITATION.cff`](CITATION.cff), or cite:

> Football Proof AI (2026). *Same-match EPL Football Prediction Model Benchmark: Elo, LightGBM, Poisson and Dixon-Coles* (Version 1.0.0). <https://footballproofai.com/research/epl-football-prediction-model-benchmark>

Preserve the version and artifact SHA-256 when the exact result matters. No DOI
has been assigned or claimed.

## Rights and source boundary

Football Proof AI's original aggregate JSON/CSV, release manifest and
documentation are offered under CC BY 4.0 only to the extent Football Proof AI
holds rights in them. Code under `scripts/`, `model/` and `tests/` is offered
under the MIT License. No raw Football-Data.co.uk rows are included,
sublicensed or asserted to be CC BY 4.0. See [`NOTICE.md`](NOTICE.md).
