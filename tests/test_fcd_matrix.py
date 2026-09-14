import json
import pickle
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np

from ConStruct.analysis import run_constraint_matrix as lower
from ConStruct.analysis import run_edge_deletion_matrix as upper
from ConStruct.analysis.fcd_matrix import (
    ValidationReferenceProfile,
    build_fcd_evaluators,
    evaluate_generated_smiles,
    fcd_details_cacheable,
    load_or_build_reference_cache,
    profile_matches,
)


def reference_bundle(profiles):
    profile_payloads = [profile.as_dict() for profile in profiles]
    statistics = {
        profile.name: {
            "profile": profile.as_dict(),
            "molecule_count": 10,
            "status": "ok",
            "error": None,
            "mu_fcd": np.zeros(2),
            "sigma_fcd": np.eye(2),
        }
        for profile in profiles
    }
    return {
        "cache_path": "/tmp/reference.npz",
        "metadata": {
            "validation_sha256": "digest",
            "fcd_version": "1.2.2",
            "profiles": profile_payloads,
            "statistics": {
                name: {
                    key: value
                    for key, value in entry.items()
                    if key not in {"mu_fcd", "sigma_fcd"}
                }
                for name, entry in statistics.items()
            },
        },
        "statistics": statistics,
    }


class ValidationReferenceTests(unittest.TestCase):
    def test_both_directions_define_all_nine_conjunctive_profiles(self):
        self.assertEqual(len(lower.VALIDATION_REFERENCE_PROFILES), 9)
        self.assertEqual(len(upper.VALIDATION_REFERENCE_PROFILES), 9)
        joint_lower = lower.VALIDATION_REFERENCE_PROFILES[4]
        joint_upper = upper.VALIDATION_REFERENCE_PROFILES[4]
        self.assertTrue(profile_matches(joint_lower, 1, 4))
        self.assertFalse(profile_matches(joint_lower, 0, 4))
        self.assertTrue(profile_matches(joint_upper, 1, 4))
        self.assertFalse(profile_matches(joint_upper, 2, 4))

    def test_reference_cache_canonicalizes_filters_reuses_and_invalidates(self):
        profiles = (
            ValidationReferenceProfile("full", None, None, "at_least"),
            ValidationReferenceProfile("ring", 1, None, "at_least"),
            ValidationReferenceProfile("ring4", 1, 4, "at_least"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "validation.pickle"
            cache = root / "reference.npz"
            with source.open("wb") as handle:
                pickle.dump(
                    {"C", "C1CC1", "C1CCC1", "not a smiles"}, handle
                )

            def predictions(_model, smiles, n_jobs):
                self.assertEqual(n_jobs, 0)
                return np.array(
                    [[float(index), float(len(smile))] for index, smile in enumerate(smiles)]
                )

            with mock.patch(
                "ConStruct.analysis.fcd_matrix.fcd.load_ref_model",
                return_value=object(),
            ), mock.patch(
                "ConStruct.analysis.fcd_matrix.fcd.get_predictions",
                side_effect=predictions,
            ) as get_predictions:
                first = load_or_build_reference_cache(source, cache, profiles)
                self.assertEqual(get_predictions.call_count, 2)
                self.assertEqual(first["statistics"]["full"]["molecule_count"], 3)
                self.assertEqual(first["statistics"]["ring"]["molecule_count"], 2)
                self.assertEqual(first["statistics"]["ring4"]["molecule_count"], 1)
                self.assertEqual(first["statistics"]["ring4"]["status"], "unavailable")
                get_predictions.reset_mock()
                load_or_build_reference_cache(source, cache, profiles)
                get_predictions.assert_not_called()

                with source.open("wb") as handle:
                    pickle.dump({"C", "CC", "C1CC1", "C1CCC1"}, handle)
                load_or_build_reference_cache(source, cache, profiles)
                self.assertEqual(get_predictions.call_count, 2)


class LiteralFCDReuseTests(unittest.TestCase):
    def test_one_evaluator_per_reference_and_81_literal_method_calls(self):
        bundle = reference_bundle(lower.VALIDATION_REFERENCE_PROFILES)

        class FakeSamplingMetrics:
            instances = []

            def __init__(self, infos, test, cfg):
                self.calls = []
                self.__class__.instances.append(self)

            def compute_fcd(self, smiles):
                self.calls.append(list(smiles))
                return {"val_sampling/fcd score": float(len(smiles))}

        with mock.patch(
            "ConStruct.analysis.fcd_matrix.SamplingMolecularMetrics",
            FakeSamplingMetrics,
        ):
            evaluators = build_fcd_evaluators(bundle, ["C", "N", "O", "F"])
            for _ in range(9):
                metrics, details = evaluate_generated_smiles(
                    ["CC", "CC", "CO"], bundle, evaluators
                )

        self.assertEqual(len(FakeSamplingMetrics.instances), 9)
        self.assertEqual(
            sum(len(instance.calls) for instance in FakeSamplingMetrics.instances),
            81,
        )
        self.assertEqual(len(metrics), 9)
        self.assertTrue(all(item["generated_molecule_count"] == 3 for item in details.values()))

    def test_missing_score_is_recorded_without_raising(self):
        profile = ValidationReferenceProfile("full", None, None, "at_least")
        bundle = reference_bundle((profile,))
        evaluator = mock.Mock()
        evaluator.compute_fcd.return_value = {"val_sampling/fcd score": None}
        metrics, details = evaluate_generated_smiles(
            ["CC", "CO"], bundle, {"full": evaluator}
        )
        self.assertIsNone(metrics["fcd_vs_validation_full"])
        self.assertEqual(details["full"]["status"], "unavailable")
        self.assertFalse(fcd_details_cacheable(details))

    def test_deterministic_tiny_cohort_null_is_cacheable(self):
        profile = ValidationReferenceProfile("full", None, None, "at_least")
        bundle = reference_bundle((profile,))
        metrics, details = evaluate_generated_smiles(
            ["CC"], bundle, {"full": mock.Mock()}
        )
        self.assertIsNone(metrics["fcd_vs_validation_full"])
        self.assertTrue(fcd_details_cacheable(details))


class FCDAggregateTests(unittest.TestCase):
    def test_writes_nine_heatmaps_grids_and_provenance_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.touch()
            bundle = reference_bundle(lower.VALIDATION_REFERENCE_PROFILES)
            cells = {}
            for generated_index, profile in enumerate(lower.PROFILES):
                fcd_details = {}
                metrics = {}
                for reference_index, reference in enumerate(
                    lower.VALIDATION_REFERENCE_PROFILES
                ):
                    key = f"fcd_vs_validation_{reference.name}"
                    score = float(generated_index + reference_index / 10)
                    metrics[key] = score
                    fcd_details[reference.name] = {
                        "metric": key,
                        "score": score,
                        "status": "ok",
                        "error": None,
                        "generated_molecule_count": 8,
                        "reference_molecule_count": 10,
                        "validation_profile": reference.as_dict(),
                    }
                cells[profile.name] = {
                    "profile": profile.name,
                    "enforced_min_rings": profile.min_rings,
                    "enforced_min_ring_length": profile.min_ring_length,
                    "num_graphs": 10,
                    "metrics": metrics,
                    "fcd": fcd_details,
                    "source_fingerprint": [{"name": "rank0.pkl"}],
                }

            lower.render_aggregate(
                root,
                cells,
                checkpoint,
                seed=0,
                requested_samples=10,
                metrics=lower.FCD_METRICS,
                reference_bundle=bundle,
            )

            payload = json.loads((root / "fcd_metrics.json").read_text())
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["unavailable_pairing_count"], 0)
            self.assertEqual(len(payload["grids"]), 9)
            self.assertEqual(len(payload["pairings"]), 9)
            self.assertEqual(
                sum(len(pairings) for pairings in payload["pairings"].values()),
                81,
            )
            for reference in lower.VALIDATION_REFERENCE_PROFILES:
                slug = f"fcd_vs_validation_{reference.name}"
                self.assertTrue((root / "grids" / f"{slug}.md").is_file())
                self.assertTrue((root / "heatmaps" / f"{slug}.png").is_file())
                self.assertTrue((root / "heatmaps" / f"{slug}.pdf").is_file())


class ManifestMigrationTests(unittest.TestCase):
    def _assert_migration(self, module, matrix_kind=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.touch()
            manifest = {
                "schema_version": 1,
                "checkpoint": str(checkpoint.resolve()),
                "experiment": "experiment",
                "seed": 0,
                "requested_samples_per_cell": 10,
                "profiles": [profile.name for profile in module.PROFILES],
                "metrics": [metric for metric, _ in module.METRICS],
                "generation_complete": True,
                "metrics_complete": True,
            }
            if matrix_kind is not None:
                manifest["matrix_kind"] = matrix_kind
            (root / "manifest.json").write_text(json.dumps(manifest))
            module.initialize_manifest(
                root,
                checkpoint,
                "experiment",
                0,
                10,
                root / "validation.pickle",
                root / "reference.npz",
            )
            upgraded = json.loads((root / "manifest.json").read_text())
            self.assertEqual(upgraded["schema_version"], 2)
            self.assertFalse(upgraded["metrics_complete"])
            self.assertEqual(upgraded["upgraded_from_schema_version"], 1)

    def test_lower_bound_manifest_migrates(self):
        self._assert_migration(lower)

    def test_upper_bound_manifest_migrates(self):
        self._assert_migration(upper, "edge_deletion_at_most")

    def test_metrics_only_upgrade_never_invokes_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.touch()
            output_root = root / "samples"
            matrix_root = output_root / "matrix" / "seed_0"
            matrix_root.mkdir(parents=True)
            legacy = {
                "schema_version": 1,
                "checkpoint": str(checkpoint.resolve()),
                "experiment": "experiment",
                "seed": 0,
                "requested_samples_per_cell": 10,
                "profiles": [profile.name for profile in lower.PROFILES],
                "metrics": [metric for metric, _ in lower.METRICS],
                "generation_complete": True,
                "metrics_complete": True,
            }
            (matrix_root / "manifest.json").write_text(json.dumps(legacy))
            bundle = reference_bundle(lower.VALIDATION_REFERENCE_PROFILES)
            evaluated_cell = {
                "fcd": {
                    profile.name: {"status": "ok"}
                    for profile in lower.VALIDATION_REFERENCE_PROFILES
                }
            }
            args = Namespace(
                repo_root=str(root),
                checkpoint=str(checkpoint),
                output_root=str(output_root),
                validation_smiles="validation.pickle",
                fcd_reference_cache="reference.npz",
                experiment="experiment",
                seed=0,
                samples=10,
                max_attempts=1,
                phase="metrics",
                force_profile=None,
            )

            with mock.patch.object(
                lower, "load_or_build_reference_cache", return_value=bundle
            ), mock.patch.object(
                lower, "build_fcd_evaluators", return_value={}
            ), mock.patch.object(
                lower, "evaluate_profile", return_value=evaluated_cell
            ) as evaluate, mock.patch.object(
                lower, "render_aggregate"
            ), mock.patch.object(
                lower, "generate_profile"
            ) as generate:
                lower.run_matrix(args)

            generate.assert_not_called()
            self.assertEqual(evaluate.call_count, 9)
            upgraded = json.loads((matrix_root / "manifest.json").read_text())
            self.assertEqual(upgraded["schema_version"], 2)
            self.assertTrue(upgraded["generation_complete"])
            self.assertTrue(upgraded["metrics_complete"])


if __name__ == "__main__":
    unittest.main()
