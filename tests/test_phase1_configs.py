import unittest
from pathlib import Path

import yaml


class Phase1ConfigTest(unittest.TestCase):
    CONFIGS = {
        "a0": Path("config/phase1_a0_control_rtx3090.yaml"),
        "a1": Path("config/phase1_a1_encoder_rtx3090.yaml"),
        "a2": Path("config/phase1_a2_decoder_rtx3090.yaml"),
        "a3": Path("config/phase1_a3_dual_rtx3090.yaml"),
    }

    def setUp(self):
        self.configs = {
            name: yaml.safe_load(path.read_text(encoding="utf-8"))
            for name, path in self.CONFIGS.items()
        }

    def test_only_adapter_placement_differs(self):
        expected = {
            "a0": (False, False),
            "a1": (True, False),
            "a2": (False, True),
            "a3": (True, True),
        }
        for name, flags in expected.items():
            model = self.configs[name]["model_overrides"]
            actual = (model["use_encoder_adapter"], model["use_decoder_adapter"])
            self.assertEqual(actual, flags)

    def test_shared_experimental_controls(self):
        first = self.configs["a0"]
        for config in self.configs.values():
            self.assertEqual(config["seed"], 7)
            self.assertEqual(config["pretrained_checkpoint"], first["pretrained_checkpoint"])
            self.assertEqual(config["selection"], first["selection"])
            self.assertEqual(config["trainer"], first["trainer"])
            self.assertEqual(config["data"], first["data"])
            for key in (
                "learning_rate", "weight_decay", "freeze_encoder_epochs",
                "adapter_bottleneck_dim", "adapter_dropout", "adapter_gate_init_bias",
            ):
                self.assertEqual(config["model_overrides"][key], first["model_overrides"][key])


if __name__ == "__main__":
    unittest.main()
