import unittest

import torch

from tamer.model.adapter import GatedBottleneckAdapter


class GatedBottleneckAdapterTest(unittest.TestCase):
    def test_initialization_is_exact_identity(self):
        adapter = GatedBottleneckAdapter(256, bottleneck_dim=64, dropout=0.1)
        adapter.train()
        x = torch.randn(2, 5, 256)
        self.assertTrue(torch.equal(adapter(x), x))

    def test_gate_and_up_projection_receive_gradients(self):
        adapter = GatedBottleneckAdapter(16, bottleneck_dim=4, dropout=0.0)
        x = torch.randn(2, 3, 16, requires_grad=True)
        adapter(x).sum().backward()
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertGreater(adapter.up.weight.grad.abs().sum().item(), 0.0)

    def test_preserves_spatial_feature_shape(self):
        adapter = GatedBottleneckAdapter(32, bottleneck_dim=8)
        x = torch.randn(2, 4, 7, 32)
        self.assertEqual(adapter(x).shape, x.shape)


if __name__ == "__main__":
    unittest.main()
