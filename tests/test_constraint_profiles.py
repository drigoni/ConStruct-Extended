import pathlib
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


class ConstraintProfileTests(unittest.TestCase):
    PROFILES = {
        "count_disabled_length_disabled": [],
        "count_1_length_disabled": [("ring_count_at_least", 1)],
        "count_2_length_disabled": [("ring_count_at_least", 2)],
        "count_disabled_length_4": [("ring_length_at_least", 4)],
        "count_disabled_length_5": [("ring_length_at_least", 5)],
        "count_1_length_4": [("ring_count_at_least", 1), ("ring_length_at_least", 4)],
        "count_1_length_5": [("ring_count_at_least", 1), ("ring_length_at_least", 5)],
        "count_2_length_4": [("ring_count_at_least", 2), ("ring_length_at_least", 4)],
        "count_2_length_5": [("ring_count_at_least", 2), ("ring_length_at_least", 5)],
    }

    def test_all_matrix_profiles_compose(self):
        config_dir = pathlib.Path(__file__).parents[1] / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            for profile, expected in self.PROFILES.items():
                with self.subTest(profile=profile):
                    cfg = compose(
                        config_name="config",
                        overrides=[
                            "+experiment=training/qm9_no_constraint_edge_addition",
                            f"+constraint={profile}",
                        ],
                    )
                    resolved = OmegaConf.to_container(cfg, resolve=True)
                    actual = []
                    for item in resolved["model"]["constraints"]:
                        field = "min_rings" if item["type"] == "ring_count_at_least" else "min_ring_length"
                        actual.append((item["type"], item[field]))
                    self.assertEqual(actual, expected)
                    self.assertEqual(resolved["model"]["use_projection"], bool(expected))
                    self.assertIn(profile, resolved["general"]["sampling_output_dir"])
                    self.assertIn("seed_0", resolved["general"]["sampling_output_dir"])

    def test_generalist_training_stops_on_sampled_validity(self):
        config_dir = pathlib.Path(__file__).parents[1] / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(
                config_name="config",
                overrides=["+experiment=training/qm9_no_constraint_edge_addition"],
            )
            resolved = OmegaConf.to_container(cfg, resolve=True)

        self.assertEqual(resolved["general"]["sample_every_val"], 1)
        self.assertGreater(resolved["general"]["samples_to_generate"], 0)
        self.assertEqual(
            resolved["train"]["early_stopping"]["monitor"],
            "val_sampling/Validity",
        )
        self.assertEqual(resolved["train"]["early_stopping"]["mode"], "max")
        self.assertEqual(resolved["train"]["early_stopping"]["min_delta"], 0.0)
        self.assertGreater(resolved["train"]["early_stopping"]["patience"], 0)


if __name__ == "__main__":
    unittest.main()
