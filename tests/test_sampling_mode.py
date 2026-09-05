import json
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import networkx as nx
import numpy as np
import torch
from omegaconf import OmegaConf

from ConStruct.diffusion_model_discrete import DiscreteDenoisingDiffusion
from ConStruct.metrics.sampling_metrics import SamplingMetrics
from ConStruct.metrics.sampling_molecular_metrics import (
    SamplingMolecularMetrics,
    _stable_frechet_distance,
)
from ConStruct.utils import PlaceHolder
from main import validate_early_stopping_config, validate_sampling_only_config


def sampling_cfg(**general_overrides):
    general = {
        "sampling_only": True,
        "test_only": "model.ckpt",
        "resume": None,
        "evaluate_all_checkpoints": False,
    }
    general.update(general_overrides)
    return OmegaConf.create(
        {
            "general": general,
            "model": {"rev_proj": "ring_count_at_most", "max_rings": 1},
            "train": {"seed": 17},
            "dataset": {"name": "qm9"},
        }
    )


class SamplingModeTests(unittest.TestCase):
    def test_sampling_metric_early_stopping_requires_every_validation(self):
        cfg = OmegaConf.create({
            "general": {
                "test_only": None,
                "sample_every_val": 1,
                "samples_to_generate": 1024,
            },
            "train": {
                "early_stopping": {
                    "enable": True,
                    "monitor": "val_sampling/Validity",
                }
            },
        })
        validate_early_stopping_config(cfg)

        cfg.general.sample_every_val = 2
        with self.assertRaisesRegex(ValueError, "sample_every_val=1"):
            validate_early_stopping_config(cfg)

        cfg.general.sample_every_val = 1
        cfg.general.samples_to_generate = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_early_stopping_config(cfg)

    def test_sampling_only_configuration_validation(self):
        validate_sampling_only_config(sampling_cfg())

        invalid_configs = [
            sampling_cfg(test_only=None),
            sampling_cfg(resume="resume.ckpt"),
            sampling_cfg(evaluate_all_checkpoints=True),
        ]
        for cfg in invalid_configs:
            with self.subTest(cfg=cfg):
                with self.assertRaises(ValueError):
                    validate_sampling_only_config(cfg)

        baseline_cfg = sampling_cfg()
        baseline_cfg.model.is_baseline = True
        with self.assertRaises(ValueError):
            validate_sampling_only_config(baseline_cfg)

    def test_projection_disabled_skips_projector_and_final_assertion(self):
        model = DiscreteDenoisingDiffusion.__new__(DiscreteDenoisingDiffusion)
        model.use_projection = False
        model.cfg = OmegaConf.create(
            {"model": {"rev_proj": "ring_count_at_least", "min_rings": 1}}
        )
        model.dataset_infos = SimpleNamespace(atom_decoder=None)

        with mock.patch(
            "ConStruct.diffusion_model_discrete.RingCountAtLeastProjector"
        ) as projector:
            self.assertIsNone(model._build_reverse_projector(object()))
            projector.assert_not_called()

        model._assert_final_constraint(object())

    def test_projection_enabled_builds_projector_and_checks_constraint(self):
        model = DiscreteDenoisingDiffusion.__new__(DiscreteDenoisingDiffusion)
        model.use_projection = True
        model.cfg = OmegaConf.create({"model": {"rev_proj": "planar"}})
        model.dataset_infos = SimpleNamespace(atom_decoder=None)
        sentinel = object()

        with mock.patch(
            "ConStruct.diffusion_model_discrete.PlanarProjector",
            return_value=sentinel,
        ) as projector:
            z_t = object()
            self.assertIs(model._build_reverse_projector(z_t), sentinel)
            projector.assert_called_once_with(z_t)

        model.cfg = OmegaConf.create(
            {"model": {"rev_proj": "ring_count_at_least", "min_rings": 1}}
        )
        path_graph = nx.to_numpy_array(nx.path_graph(4), dtype=int)
        final_batch = SimpleNamespace(
            E=torch.tensor(path_graph).unsqueeze(0),
            node_mask=torch.ones((1, 4), dtype=torch.bool),
        )
        with self.assertRaises(AssertionError):
            model._assert_final_constraint(final_batch)

    def test_sampling_hooks_skip_likelihood_and_wandb_setup(self):
        model = DiscreteDenoisingDiffusion.__new__(DiscreteDenoisingDiffusion)
        model.cfg = sampling_cfg()
        model.test_sampling_metrics = mock.Mock()

        with mock.patch(
            "ConStruct.diffusion_model_discrete.utils.setup_wandb"
        ) as setup_wandb:
            model.on_test_epoch_start()
            setup_wandb.assert_not_called()

        model.test_sampling_metrics.reset.assert_called_once_with()
        self.assertIsNone(model.test_step(data=object(), i=0))

    def test_json_metrics_do_not_construct_wandb_histograms(self):
        scalar_metric = mock.Mock()
        scalar_metric.compute.return_value = torch.tensor(1.0)
        scalar_metric.device = torch.device("cpu")
        degree_metric = mock.Mock()
        histogram = (np.array([1.0]), np.array([0.0, 1.0]))
        degree_metric.get_hists_to_log.return_value = (
            histogram,
            histogram,
            histogram,
        )
        harness = SimpleNamespace(
            stat=SimpleNamespace(
                num_nodes=None,
                atom_types=None,
                bond_types=None,
                degree_hist=[np.array([1.0])],
            ),
            test=True,
            cfg=OmegaConf.create({
                "model": {
                    "rev_proj": None,
                    "constraints": [
                        {"type": "ring_count_at_least", "min_rings": 1},
                        {"type": "ring_length_at_least", "min_ring_length": 4},
                    ],
                }
            }),
            num_nodes_w1=scalar_metric,
            node_types_tv=scalar_metric,
            edge_types_tv=scalar_metric,
            disconnected=scalar_metric,
            mean_components=scalar_metric,
            max_components=scalar_metric,
            mean_planarity=scalar_metric,
            mean_no_cycles=scalar_metric,
            mean_lobster_components=scalar_metric,
            mean_ring_count_satisfaction=scalar_metric,
            mean_ring_length_satisfaction=scalar_metric,
            mean_joint_constraint_satisfaction=scalar_metric,
            deg_histogram=degree_metric,
            domain_metrics=None,
        )

        patches = [
            mock.patch(
                "ConStruct.metrics.sampling_metrics.number_nodes_distance",
                return_value=torch.tensor(0.0),
            ),
            mock.patch(
                "ConStruct.metrics.sampling_metrics.node_types_distance",
                return_value=(torch.tensor(0.0), None),
            ),
            mock.patch(
                "ConStruct.metrics.sampling_metrics.bond_types_distance",
                return_value=(torch.tensor(0.0), None),
            ),
            mock.patch(
                "ConStruct.metrics.sampling_metrics.connected_components",
                return_value=torch.tensor([1.0]),
            ),
            mock.patch(
                "ConStruct.metrics.sampling_metrics.planarity_ratio",
                return_value=torch.tensor([1.0]),
            ),
            mock.patch(
                "ConStruct.metrics.sampling_metrics.no_cycles_ratio",
                return_value=torch.tensor([1.0]),
            ),
            mock.patch(
                "ConStruct.metrics.sampling_metrics.lobster_components_ratio",
                return_value=torch.tensor([1.0]),
            ),
            mock.patch(
                "ConStruct.metrics.sampling_metrics.ring_count_at_least_satisfaction_ratio",
                return_value=torch.tensor([1.0]),
            ),
            mock.patch(
                "ConStruct.metrics.sampling_metrics.ring_length_at_least_satisfaction_ratio",
                return_value=torch.tensor([1.0]),
            ),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8], mock.patch(
            "ConStruct.metrics.sampling_metrics.wandb.run", object()
        ), mock.patch(
            "ConStruct.metrics.sampling_metrics.wandb.Histogram"
        ) as histogram_constructor, mock.patch(
            "ConStruct.metrics.sampling_metrics.wandb.log"
        ) as wandb_log:
            metrics, _ = SamplingMetrics.compute_all_metrics(
                harness,
                generated_graphs=[],
                current_epoch=0,
                local_rank=1,
                log_to_wandb=False,
            )

        histogram_constructor.assert_not_called()
        wandb_log.assert_not_called()
        self.assertEqual(
            metrics["test_sampling/generated_deg_hist"]["counts"], [1.0]
        )
        self.assertEqual(metrics["test_sampling/ring_count_satisfaction"], 1.0)
        self.assertEqual(metrics["test_sampling/ring_length_satisfaction"], 1.0)
        self.assertEqual(metrics["test_sampling/joint_constraint_satisfaction"], 1.0)

    def test_sampling_artifacts_include_pickles_and_metric_metadata(self):
        with tempfile.TemporaryDirectory() as output_dir:
            cfg = sampling_cfg()
            cfg.general.sampling_output_dir = output_dir
            cfg.general.final_model_samples_to_generate = 1
            cfg.general.final_model_samples_to_save = 0
            cfg.general.final_model_chains_to_save = 0
            cfg.general.faster_sampling = 1

            batch = PlaceHolder(
                X=torch.zeros((1, 2), dtype=torch.long),
                charges=torch.zeros((1, 2), dtype=torch.long),
                E=torch.zeros((1, 2, 2), dtype=torch.long),
                y=None,
                node_mask=torch.ones((1, 2), dtype=torch.bool),
            )
            metrics = {
                "test_sampling/Validity": 1.0,
                "test_sampling/generated_deg_hist": {
                    "counts": [1.0],
                    "bins": [0.0, 1.0],
                },
            }
            metric_module = mock.Mock()
            metric_module.compute_all_metrics.return_value = (metrics, None)
            strategy = mock.Mock()
            trainer = SimpleNamespace(world_size=1, num_devices=1, strategy=strategy)
            harness = SimpleNamespace(
                cfg=cfg,
                trainer=trainer,
                _trainer=trainer,
                global_rank=0,
                local_rank=0,
                current_epoch=3,
                use_projection=False,
                dataset_infos=SimpleNamespace(),
                test_sampling_metrics=metric_module,
                print=lambda *args, **kwargs: None,
                sample_n_graphs=mock.Mock(return_value=[batch]),
                _constraint_target=lambda: ("ring_count_at_most", 1),
                _json_ready=DiscreteDenoisingDiffusion._json_ready,
            )

            DiscreteDenoisingDiffusion._run_final_sampling(
                harness, sampling_only=True
            )

            pickle_path = os.path.join(
                output_dir, "generated_samples_rank0.pkl"
            )
            metrics_path = os.path.join(output_dir, "sampling_metrics.json")
            self.assertTrue(os.path.exists(pickle_path))
            self.assertTrue(os.path.exists(metrics_path))
            with open(pickle_path, "rb") as handle:
                self.assertEqual(len(pickle.load(handle)), 1)
            with open(metrics_path) as handle:
                payload = json.load(handle)

            self.assertFalse(payload["metadata"]["projection_enabled"])
            self.assertEqual(payload["metadata"]["seed"], 17)
            self.assertEqual(payload["metadata"]["generated_samples"], 1)
            self.assertEqual(
                payload["metadata"]["constraints"],
                [{"type": "ring_count_at_most", "max_rings": 1}],
            )
            self.assertEqual(
                payload["metrics"]["test_sampling/generated_deg_hist"]["counts"],
                [1.0],
            )
            metric_module.compute_all_metrics.assert_called_once()
            self.assertFalse(
                metric_module.compute_all_metrics.call_args.kwargs[
                    "log_to_wandb"
                ]
            )


class FCDStabilityTests(unittest.TestCase):
    def test_stable_frechet_distance_matches_diagonal_gaussians(self):
        distance = _stable_frechet_distance(
            np.zeros(2), np.eye(2), np.zeros(2), 4 * np.eye(2)
        )
        self.assertAlmostEqual(distance, 2.0)

    def test_stable_frechet_distance_handles_near_singular_covariance(self):
        covariance = np.array([[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]])
        distance = _stable_frechet_distance(
            np.zeros(2), covariance, np.zeros(2), np.eye(2)
        )
        self.assertTrue(np.isfinite(distance))
        self.assertGreaterEqual(distance, 0.0)

    def test_stable_frechet_distance_handles_high_dimension(self):
        rng = np.random.default_rng(7)
        left = rng.normal(size=(128, 128))
        right = rng.normal(size=(128, 128))
        distance = _stable_frechet_distance(
            np.zeros(128),
            np.einsum("ik,jk->ij", left, left, optimize=False) / 128,
            np.zeros(128),
            np.einsum("ik,jk->ij", right, right, optimize=False) / 128,
        )
        self.assertTrue(np.isfinite(distance))
        self.assertGreaterEqual(distance, 0.0)

    def test_fcd_disables_workers_and_falls_back_to_psd_calculation(self):
        harness = SimpleNamespace(
            test=True,
            val_fcd_mu=np.zeros(2),
            val_fcd_sigma=np.eye(2),
        )
        activations = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

        with mock.patch(
            "ConStruct.metrics.sampling_molecular_metrics.fcd.load_ref_model",
            return_value=object(),
        ), mock.patch(
            "ConStruct.metrics.sampling_molecular_metrics.fcd.get_predictions",
            return_value=activations,
        ) as predictions, mock.patch(
            "ConStruct.metrics.sampling_molecular_metrics.fcd.calculate_frechet_distance",
            side_effect=ValueError("Imaginary component"),
        ):
            result = SamplingMolecularMetrics.compute_fcd(
                harness, ["CC", "CCC", "CO"]
            )

        self.assertGreaterEqual(result["test_sampling/fcd score"], 0.0)
        self.assertEqual(predictions.call_args.kwargs["n_jobs"], 0)

    def test_invalid_cached_fcd_does_not_create_ratio_one(self):
        harness = SimpleNamespace(
            test=True,
            dataset_infos=SimpleNamespace(is_molecular=True, is_tls=False),
        )
        ratios = SamplingMetrics.compute_ratios_to_ref(
            harness,
            reference_metrics={"val_sampling/fcd score": -1},
            generated_metrics={"test_sampling/fcd score": -1},
        )
        self.assertIsNone(ratios["test_ratio/fcd score"])
        self.assertIsNone(ratios["test_ratio/average"])

    def test_json_ready_replaces_non_finite_numbers(self):
        result = DiscreteDenoisingDiffusion._json_ready(
            {
                "nan": float("nan"),
                "numpy": np.array([np.float64("inf")]),
                "tensor": torch.tensor(float("nan")),
            }
        )
        self.assertEqual(
            result, {"nan": None, "numpy": [None], "tensor": None}
        )


if __name__ == "__main__":
    unittest.main()
