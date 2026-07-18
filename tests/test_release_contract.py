# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "epl-football-prediction-model-benchmark-1.0.0.json": (
        ROOT / "data/epl-football-prediction-model-benchmark-1.0.0.json",
        "60eedfeca8f0e8b0bad76f65734fbd89797f5246910aea4a2e86553857d1cde1",
    ),
    "epl-football-prediction-model-benchmark-1.0.0.csv": (
        ROOT / "data/epl-football-prediction-model-benchmark-1.0.0.csv",
        "ecdbdf708c649791de3f118798c110e0ca9c2cbc5c026760e2634d5d4e0e746a",
    ),
    "epl-football-prediction-model-benchmark-1.0.0-manifest.json": (
        ROOT / "data/epl-football-prediction-model-benchmark-1.0.0-manifest.json",
        "8037b509f57984b5c076ace79917a5c6868a3136e82d6737fbdbb1a6635a0e9b",
    ),
    "generate-epl-football-prediction-model-benchmark-1.0.0.py": (
        ROOT / "scripts/generate-epl-football-prediction-model-benchmark.py",
        "ae61549bf96803665660b153a20a973d29faa18d17e982e344a9c99ff9dd06a3",
    ),
    "football-1x2-empirical-benchmark-1.0.0-manifest.json": (
        ROOT / "data/football-1x2-empirical-benchmark-1.0.0-manifest.json",
        "801f8c6c23e30f945dcba38e4c256fd4e9f3652f7e2ea965e4788f0e7cf9014a",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseContractTest(unittest.TestCase):
    def test_release_assets_and_checksum_index_are_exact(self) -> None:
        expected_index = {
            name: expected_hash for name, (_, expected_hash) in FILES.items()
        }
        observed_index = {}
        for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            checksum, filename = line.split("  ", 1)
            observed_index[filename] = checksum
        self.assertEqual(observed_index, expected_index)
        for path, expected_hash in FILES.values():
            self.assertEqual(digest(path), expected_hash, path)

    def test_benchmark_uses_one_complete_fixture_set_and_stays_fail_closed(self) -> None:
        artifact_path = FILES["epl-football-prediction-model-benchmark-1.0.0.json"][0]
        artifact_bytes = artifact_path.read_bytes()
        artifact = json.loads(artifact_bytes)
        self.assertNotIn(b"\n", artifact_bytes)
        self.assertEqual(artifact["evaluation"]["evaluationMatchCount"], 1_520)
        self.assertEqual(artifact["evaluation"]["evaluationWeekCount"], 143)
        self.assertTrue(artifact["evaluation"]["allModelsFullCoverage"])
        self.assertEqual(len(artifact["leaderboard"]), 10)
        fixture_hashes = {row["fixtureKeySetSha256"] for row in artifact["leaderboard"]}
        self.assertEqual(
            fixture_hashes,
            {artifact["evaluation"]["evaluationFixtureKeySetSha256"]},
        )
        self.assertTrue(all(row["n"] == 1_520 for row in artifact["leaderboard"]))
        self.assertFalse(artifact["pointInTimeBoundary"]["productionActivationEligible"])
        self.assertFalse(artifact["claimsBoundary"]["profitClaim"])
        self.assertFalse(artifact["claimsBoundary"]["bettingAdvice"])
        self.assertFalse(artifact["source"]["rawRowsRedistributed"])

    def test_manifest_binds_sources_generator_and_model_dependencies(self) -> None:
        manifest_path = FILES[
            "epl-football-prediction-model-benchmark-1.0.0-manifest.json"
        ][0]
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        self.assertEqual(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode(),
            manifest_bytes,
        )
        self.assertEqual(len(manifest["sourceInputs"]), 10)
        self.assertTrue(manifest["rawSourceLicenseNotGranted"])
        self.assertFalse(manifest["rawRowsRedistributed"])
        self.assertEqual(
            manifest["sourceManifest"]["sha256"],
            digest(ROOT / manifest["sourceManifest"]["path"]),
        )
        for dependency in manifest["dependencyFiles"]:
            self.assertEqual(
                dependency["sha256"],
                digest(ROOT / dependency["repositoryPath"]),
            )
        for required_module in (
            "__init__.py",
            "domain.py",
            "features.py",
            "training.py",
            "metrics.py",
        ):
            self.assertTrue((ROOT / "model/proofxi_ml" / required_module).is_file())

    def test_archive_contains_no_raw_source_csv(self) -> None:
        raw_paths = [path for path in ROOT.rglob("E0.csv") if ".git" not in path.parts]
        self.assertEqual(raw_paths, [])

    def test_discovery_and_citation_metadata_keep_the_canonical_boundary(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        canonical = (
            "https://footballproofai.com/research/"
            "epl-football-prediction-model-benchmark"
        )
        self.assertIn(canonical, readme[:900])
        self.assertIn(canonical, citation)
        self.assertNotIn("doi:", citation.lower())
        self.assertEqual(zenodo["version"], "1.0.0")

    def test_license_metadata_is_complete_and_has_no_template_placeholders(self) -> None:
        mit = (ROOT / "LICENSES/MIT.txt").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2026 Football Proof AI", mit)
        inspected = (
            "README.md",
            "NOTICE.md",
            "LICENSE",
            "LICENSES/CC-BY-4.0.txt",
            "LICENSES/MIT.txt",
            "CITATION.cff",
            ".zenodo.json",
        )
        placeholders = (
            "<year>",
            "<copyright holders>",
            "PLACEHOLDER",
            "CHANGEME",
        )
        for relative_path in inspected:
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            for placeholder in placeholders:
                self.assertNotIn(placeholder, text, relative_path)


if __name__ == "__main__":
    unittest.main()
