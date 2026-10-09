import itertools
import json
import os
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import networkx as nx
import numpy as np
import torch

from ConStruct.analysis.final_suite_telemetry import FinalSuiteTelemetry
from ConStruct.analysis.run_final_constraint_suite import (
    PROFILES,
    FCDContext,
    Profile,
    balanced_batch_sizes,
    cross_constraint_validity,
    exact_histograms,
    generation_batch_sizes,
    generation_command,
    generation_complete,
    generate_cell,
    metrics_cell,
    parse_args,
    partition_metric_profiles,
    profile_run_lock,
    select_checkpoint,
    seed_run_lock,
    stuck_payload,
    write_manifest,
    _deletion_reachability,
    _insertion_reachability,
    _stuck_graph,
)
from ConStruct.projector.graph_cycles import (
    bitmask_cycle_summary,
    count_simple_cycles,
    max_simple_cycle_length,
)
from ConStruct.utils import PlaceHolder


def one_hot_state(edge_classes):
    edge_classes = torch.as_tensor(edge_classes, dtype=torch.long).unsqueeze(0)
    n = edge_classes.shape[-1]
    X = torch.nn.functional.one_hot(torch.zeros((1, n), dtype=torch.long), 4).float()
    charges = torch.nn.functional.one_hot(torch.ones((1, n), dtype=torch.long), 3).float()
    E = torch.nn.functional.one_hot(edge_classes, 5).float()
    return PlaceHolder(X=X, charges=charges, E=E, y=None, node_mask=torch.ones((1, n), dtype=torch.bool))


class FinalConstraintSuiteTests(unittest.TestCase):
    def test_grid_directions_aliases_and_900_pairs(self):
        self.assertEqual(len(PROFILES), 30)
        self.assertEqual(len({profile.name for profile in PROFILES}), 30)
        noop = Profile(0, 0)
        self.assertEqual(noop.constraints("insertion"), [])
        self.assertEqual(noop.effective_values("insertion"), (None, None))
        self.assertEqual(
            noop.constraints("deletion"),
            [
                {"type": "ring_count_at_most", "max_rings": 0},
                {"type": "ring_length_at_most", "max_ring_length": 0},
            ],
        )
        rows = {profile.name: cross_constraint_validity("insertion", [0], [0]) for profile in PROFILES}
        self.assertEqual(sum(len(row) for row in rows.values()), 900)
        self.assertEqual(rows[PROFILES[0].name][PROFILES[0].name]["percentage"], 100.0)

    def test_exact_histogram_bins(self):
        histograms = exact_histograms(list(range(11)), list(range(10)) + [9])
        self.assertEqual(list(histograms["cycle_count"]), [str(i) for i in range(10)] + [">=10"])
        self.assertEqual(histograms["cycle_count"][">=10"], 1)
        self.assertEqual(list(histograms["max_cycle_length"]), [str(i) for i in range(10)])

    def test_balanced_batch_sizes(self):
        self.assertEqual(balanced_batch_sizes(10000, 2), [5000, 5000])
        self.assertEqual(balanced_batch_sizes(10000, 3), [3334, 3333, 3333])
        self.assertEqual(balanced_batch_sizes(10000, 4), [2500] * 4)

    def test_generation_batch_sizes_start_at_requested_cap_then_back_off(self):
        self.assertEqual(generation_batch_sizes(10000, 10000, 1), [10000])
        self.assertEqual(generation_batch_sizes(10000, 10000, 2), [5000, 5000])
        self.assertEqual(generation_batch_sizes(10000, 10000, 3), [3334, 3333, 3333])
        self.assertEqual(generation_batch_sizes(10000, 5000, 1), [5000, 5000])

    def test_seed_run_lock_rejects_duplicate_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with seed_run_lock(root, 1, "generate") as lock_path:
                self.assertTrue(lock_path.is_file())
                with self.assertRaisesRegex(RuntimeError, "already holds"):
                    with seed_run_lock(root, 1, "metrics"):
                        pass
                with seed_run_lock(root, 2, "generate"):
                    pass
            with seed_run_lock(root, 1, "metrics"):
                pass

    def test_profile_locks_allow_disjoint_shards_and_reject_same_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = PROFILES[0].name
            second = PROFILES[15].name
            with profile_run_lock(root, 1, "generate", first):
                with profile_run_lock(root, 1, "generate", second):
                    pass
                with self.assertRaisesRegex(RuntimeError, "already holds"):
                    with profile_run_lock(root, 1, "generate", first):
                        pass

    def test_metric_scan_keeps_ready_cells_after_an_incomplete_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch=1.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            args = SimpleNamespace(
                modality="deletion", seed=1, samples=10000, trajectory_samples=256,
            )
            profiles = [PROFILES[0], PROFILES[1], PROFILES[15]]

            def ready(cell, identity):
                del identity
                return cell.parent.name in {PROFILES[0].name, PROFILES[15].name}

            with mock.patch(
                "ConStruct.analysis.run_final_constraint_suite.generation_complete",
                side_effect=ready,
            ):
                complete, incomplete = partition_metric_profiles(
                    args, root, checkpoint, profiles
                )
            self.assertEqual(complete, [PROFILES[0], PROFILES[15]])
            self.assertEqual(incomplete, [PROFILES[1]])

    def test_incomplete_metric_cell_is_skipped_without_context_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch=1.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            args = SimpleNamespace(
                modality="insertion", seed=2, samples=10000,
                trajectory_samples=256, force_profile=[],
            )
            with mock.patch(
                "ConStruct.analysis.run_final_constraint_suite.generation_complete",
                return_value=False,
            ):
                result = metrics_cell(
                    args, root, root, PROFILES[0], checkpoint, None, None
                )
            self.assertFalse(result)

    def test_profile_range_validation(self):
        args = parse_args([
            "--modality", "deletion", "--seed", "1",
            "--profile-start-index", "15", "--profile-end-index", "30",
        ])
        self.assertEqual((args.profile_start_index, args.profile_end_index), (15, 30))
        with self.assertRaises(SystemExit):
            parse_args([
                "--modality", "deletion", "--seed", "1",
                "--profile-start-index", "15", "--profile-end-index", "15",
            ])

    def test_checkpoint_selection_ignores_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "checkpoints/qm9_no_constraint_edge_absorbing_large_seed_0"
            target.mkdir(parents=True)
            for name in ("epoch=9.ckpt", "epoch=101.ckpt", "last-v9.ckpt", "epoch=bad.ckpt"):
                (target / name).touch()
            self.assertEqual(select_checkpoint(root, "deletion", 0).name, "epoch=101.ckpt")

    def test_generation_command_is_strict_and_direct(self):
        command = generation_command(
            Path("/repo"), "deletion", 1, Profile(2, 4), Path("/checkpoints/epoch=679.ckpt"),
            Path("/out"), 10000, [5000, 5000], 17,
        )
        joined = " ".join(command)
        self.assertIn("strict_generation=true", joined)
        self.assertIn("trajectory_samples=17", joined)
        self.assertIn("ring_count_at_most", joined)
        self.assertIn("general.test_only='/checkpoints/epoch=679.ckpt'", joined)
        self.assertNotIn("+constraint=", joined)

    def test_cuda_oom_restarts_cell_with_more_balanced_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch=1.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "samples"
            args = SimpleNamespace(
                modality="deletion", seed=0, samples=10000, trajectory_samples=0,
                force_profile=[], max_attempts=3, initial_generation_batch_size=5000,
                device="cpu",
            )
            calls = []
            def fake_run(command, **kwargs):
                calls.append(command)
                log_handle = kwargs["stdout"]
                self.assertTrue(Path(log_handle.name).is_file())
                self.assertIn("Command:", Path(log_handle.name).read_text())
                if len(calls) == 1:
                    log_handle.write("CUDA out of memory\n")
                    log_handle.flush()
                    return SimpleNamespace(returncode=1)
                output_arg = next(value for value in command if value.startswith("general.sampling_output_dir="))
                work = Path(output_arg.split("=", 1)[1].strip("'"))
                work.mkdir(parents=True)
                for name in ("generated_samples_rank0.pkl", "generation_telemetry.npz", "blocked_actions.json"):
                    (work / name).write_bytes(b"x")
                log_handle.write("ok\n")
                log_handle.flush()
                return SimpleNamespace(returncode=0)
            with mock.patch("ConStruct.analysis.run_final_constraint_suite.subprocess.run", side_effect=fake_run):
                generate_cell(args, root, output, Profile(None, None), checkpoint, None)
            self.assertEqual(len(calls), 2)
            manifest = json.loads((output / Profile(None, None).name / "seed_0" / "generation_manifest.json").read_text())
            self.assertEqual(manifest["batch_sizes"], [3334, 3333, 3333])
            first_log = output / "errors" / "seed_0" / f"{Profile(None, None).name}_attempt_1.log"
            self.assertTrue(first_log.is_file())
            self.assertIn("CUDA out of memory", first_log.read_text())

    def test_interrupted_cell_restarts_and_completed_cell_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch=1.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "samples"
            profile = Profile(None, None)
            args = SimpleNamespace(
                modality="deletion", seed=0, samples=10, trajectory_samples=0,
                force_profile=[], max_attempts=1, initial_generation_batch_size=10,
                device="cpu",
            )

            def work_from(command):
                output_arg = next(
                    value
                    for value in command
                    if value.startswith("general.sampling_output_dir=")
                )
                return Path(output_arg.split("=", 1)[1].strip("'"))

            def interrupt_run(command, **kwargs):
                work = work_from(command)
                work.mkdir(parents=True)
                (work / "partial").write_text("incomplete")
                raise KeyboardInterrupt

            with mock.patch(
                "ConStruct.analysis.run_final_constraint_suite.subprocess.run",
                side_effect=interrupt_run,
            ), self.assertRaises(KeyboardInterrupt):
                generate_cell(args, root, output, profile, checkpoint, None)

            cell = output / profile.name / "seed_0"
            self.assertFalse((cell / "generation_manifest.json").exists())

            def successful_run(command, **kwargs):
                work = work_from(command)
                self.assertFalse((work / "partial").exists())
                work.mkdir(parents=True, exist_ok=True)
                for name in ("generated_samples_rank0.pkl", "generation_telemetry.npz", "blocked_actions.json"):
                    (work / name).write_bytes(b"x")
                return SimpleNamespace(returncode=0)

            with mock.patch(
                "ConStruct.analysis.run_final_constraint_suite.subprocess.run",
                side_effect=successful_run,
            ):
                generate_cell(args, root, output, profile, checkpoint, None)
            self.assertTrue((cell / "generation_manifest.json").is_file())

            with mock.patch("ConStruct.analysis.run_final_constraint_suite.subprocess.run") as run:
                generate_cell(args, root, output, profile, checkpoint, None)
                run.assert_not_called()

    def test_manifest_reuse_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory)
            checkpoint = cell / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            identity = {
                "schema_version": 1,
                "modality": "deletion",
                "reverse_operation": "insertion",
                "seed": 0,
                "profile": "test",
                "labeled_constraints": {},
                "effective_constraints": [],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": "unused",
                "requested_samples": 1,
                "trajectory_samples": 1,
            }
            for name in ("generated_samples_rank0.pkl", "generation_telemetry.npz", "blocked_actions.json"):
                (cell / name).write_bytes(b"x")
            write_manifest(cell, identity, [1])
            self.assertTrue(generation_complete(cell, identity))
            with self.assertRaises(RuntimeError):
                generation_complete(cell, {**identity, "requested_samples": 2})

    def test_cycle_bitmask_matches_existing_enumerator(self):
        rng = random.Random(7)
        for n in range(1, 10):
            pairs = list(itertools.combinations(range(n), 2))
            for _ in range(40):
                bits = rng.randrange(1 << len(pairs))
                graph = nx.Graph()
                graph.add_nodes_from(range(n))
                graph.add_edges_from(edge for index, edge in enumerate(pairs) if bits & (1 << index))
                self.assertEqual(
                    bitmask_cycle_summary(n, bits),
                    (count_simple_cycles(graph), max_simple_cycle_length(graph)),
                )

    def test_edge_changes_ignore_positive_class_swaps_and_count_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            before = one_hot_state([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
            raw = one_hot_state([[0, 2, 1], [2, 0, 0], [1, 0, 0]])
            after = one_hot_state([[0, 2, 0], [2, 0, 0], [0, 0, 0]])
            collector = FinalSuiteTelemetry(directory, 1, 1, 4, "insertion", 2, collapse_charges=torch.tensor([-1, 0, 1]))
            collector.begin_batch(before, 0)
            collector.record_transition(before, raw, after, 1, blocked_attempts=1, newly_blocked_pairs=1)
            collector.record_transition(after, raw, after, 0, blocked_attempts=1, newly_blocked_pairs=0)
            collector.end_batch()
            collector.finish()
            payload = json.loads((Path(directory) / "blocked_actions.json").read_text())
            self.assertEqual(payload["totals"]["blocked_attempts"], 2)
            self.assertEqual(payload["totals"]["newly_blocked_graph_edge_pairs"], 1)
            self.assertEqual(payload["totals"]["repeated_blocked_attempts"], 1)
            self.assertEqual(payload["totals"]["edge_additions"], 0)

    def test_trajectory_cohort_is_seed_deterministic(self):
        first = FinalSuiteTelemetry("/tmp/a", 10000, 256, 99, "insertion", 500)
        second = FinalSuiteTelemetry("/tmp/b", 10000, 256, 99, "insertion", 500)
        third = FinalSuiteTelemetry("/tmp/c", 10000, 256, 100, "insertion", 500)
        np.testing.assert_array_equal(first.cohort_indices, second.cohort_indices)
        self.assertFalse(np.array_equal(first.cohort_indices, third.cohort_indices))

    def test_reverse_deletion_reachability_is_exact_and_budgeted(self):
        atoms = np.zeros(6, dtype=int)
        charges = np.zeros(6, dtype=int)
        edges = np.zeros((6, 6), dtype=int)
        edges[0, 1:] = edges[1:, 0] = 1
        edges[1, 2] = edges[2, 1] = 1
        profile = Profile(1, None)

        result = _deletion_reachability(atoms, charges, edges, profile, 100)
        self.assertEqual(result["status"], "repairable")
        self.assertLessEqual(result["witness"]["maximum_degree"], 4)
        self.assertGreaterEqual(result["witness"]["cycle_count"], 1)

        exhausted = _deletion_reachability(atoms, charges, edges, profile, 1)
        self.assertEqual(exhausted["status"], "indeterminate")
        self.assertEqual(exhausted["reason"], "search_budget_exhausted")

    def test_reverse_insertion_reachability_uses_immutable_bond_valence(self):
        atoms = np.zeros(3, dtype=int)
        charges = np.zeros(3, dtype=int)
        overvalent = np.zeros((3, 3), dtype=int)
        overvalent[0, 1:] = overvalent[1:, 0] = 3
        result = _insertion_reachability(
            atoms, charges, overvalent, Profile(9, 9), 100
        )
        self.assertEqual(result["status"], "stuck")
        self.assertEqual(result["maximum_bond_valence"], 6.0)

        repairable = np.zeros((3, 3), dtype=int)
        repairable[0, 1:] = repairable[1:, 0] = 2
        result = _insertion_reachability(
            atoms, charges, repairable, Profile(9, 9), 100
        )
        self.assertEqual(result["status"], "repairable")

    def test_dead_end_onset_uses_endpoint_first_binary_search(self):
        state_count = 21
        timesteps = np.arange(500, -1, -25)
        X = np.zeros((state_count, 1), dtype=int)
        X[:, 0] = np.arange(state_count)
        charges = np.zeros_like(X)
        edges = np.zeros((state_count, 1, 1), dtype=int)
        masks = np.ones_like(X, dtype=bool)

        def fake_reachability(modality, profile, atoms, state_charges, state_edges, budget):
            del modality, profile, state_charges, state_edges, budget
            stuck = int(atoms[0]) >= 13
            return {
                "status": "stuck" if stuck else "repairable",
                "reason": "synthetic",
                "currently_valid": False,
                "states_explored": 1,
                "states_seen": 1,
                "constraint_prunes": 0,
            }

        task = (
            7, timesteps, X, charges, edges, masks, list(range(state_count)),
            "insertion", Profile(1, None), 100,
        )
        with mock.patch(
            "ConStruct.analysis.run_final_constraint_suite._state_reachability",
            side_effect=fake_reachability,
        ) as reachability:
            result = _stuck_graph(task)
        self.assertEqual(result["first_stuck_selected_position"], 13)
        self.assertEqual(result["first_stuck_timestep"], 175)
        self.assertLessEqual(reachability.call_count, 1 + int(np.ceil(np.log2(state_count))))
        self.assertFalse(result["evaluations"][12]["stuck"])
        self.assertTrue(result["evaluations"][13]["stuck"])

    def test_unconstrained_profile_is_explicitly_not_applicable(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry = Path(directory) / "telemetry.npz"
            np.savez_compressed(
                telemetry,
                global_indices=np.asarray([3]),
                timesteps=np.asarray([25, 0]),
                X=np.zeros((2, 1, 1), dtype=np.int8),
                charges=np.zeros((2, 1, 1), dtype=np.int8),
                E=np.zeros((2, 1, 1, 1), dtype=np.int8),
                node_mask=np.ones((2, 1, 1), dtype=bool),
            )
            with mock.patch(
                "ConStruct.analysis.run_final_constraint_suite._stuck_graph"
            ) as graph_check:
                payload = stuck_payload(
                    telemetry, "insertion", Profile(None, None), 25, 1, 100
                )
            graph_check.assert_not_called()
            self.assertEqual(payload["status"], "not_applicable")
            self.assertEqual(payload["reason"], "no_effective_projection_constraints")

    def test_final_valid_endpoint_skips_all_earlier_dead_end_checks(self):
        timesteps = np.asarray([50, 25, 0])
        X = np.zeros((3, 1), dtype=int)
        charges = np.zeros_like(X)
        edges = np.zeros((3, 1, 1), dtype=int)
        masks = np.ones_like(X, dtype=bool)
        task = (
            8, timesteps, X, charges, edges, masks, [0, 1, 2],
            "insertion", Profile(1, None), 100,
        )
        valid = {
            "status": "repairable",
            "reason": "currently_valid",
            "currently_valid": True,
            "states_explored": 0,
            "states_seen": 0,
            "constraint_prunes": 0,
        }
        with mock.patch(
            "ConStruct.analysis.run_final_constraint_suite._state_reachability",
            return_value=valid,
        ) as reachability:
            result = _stuck_graph(task)
        self.assertEqual(reachability.call_count, 1)
        self.assertEqual(result["status"], "skipped_final_valid")
        self.assertTrue(all(not entry["stuck"] for entry in result["evaluations"]))

    def test_generated_fcd_one_call_per_fingerprint_and_cache_reuse(self):
        context = FCDContext.__new__(FCDContext)
        context.model = object()
        context.device = "cpu"
        context.workers = 1
        reference = type("Reference", (), {"name": "full"})()
        context.references = (reference,)
        context.bundle = {"statistics": {"full": {
            "status": "ok", "error": None, "molecule_count": 3,
            "mu_fcd": np.zeros(2), "sigma_fcd": np.eye(2),
        }}}
        activations = np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "ConStruct.analysis.run_final_constraint_suite.fcd.get_predictions",
            return_value=activations,
        ) as predictions:
            context.scores(["C", "CC", "CCC"], "abc", Path(directory))
            context.scores(["C", "CC", "CCC"], "abc", Path(directory))
            self.assertEqual(predictions.call_count, 1)
            context.scores(["C", "CC", "CCC"], "def", Path(directory))
            self.assertEqual(predictions.call_count, 2)

    def test_wrapper_interfaces(self):
        root = Path(__file__).resolve().parents[1]
        for name, modality in (("run_final_insertion.sh", "insertion"), ("run_final_deletion.sh", "deletion")):
            text = (root / "scripts" / name).read_text()
            self.assertIn(f"--modality {modality}", text)
            self.assertIn('--seed "$1"', text)
            self.assertIn('--phase "$2"', text)
            self.assertIn("FINAL_STUCK_MAX_SEARCH_STATES", text)
            self.assertIn("FINAL_PROFILE_START", text)
            self.assertIn("FINAL_PROFILE_END", text)


if __name__ == "__main__":
    unittest.main()
