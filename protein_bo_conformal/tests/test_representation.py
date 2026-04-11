"""Unit tests for Day 4 representation infrastructure."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from representation.interface import build_encoder


class RepresentationTests(unittest.TestCase):
    def test_onehot_flattened_shape_and_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "name": "onehot",
                "cache_dir": tmp_dir,
                "cache_enabled": True,
                "batch_size": 4,
                "onehot_mode": "flattened",
                "fixed_length": 4,
            }
            encoder = build_encoder(config)
            batch = encoder.encode(["ACDX", "AAAA"])
            single = np.vstack([encoder.encode(["ACDX"])[0], encoder.encode(["AAAA"])[0]])
            self.assertEqual(batch.shape, (2, 84))
            self.assertTrue(np.allclose(batch, single))

    def test_onehot_bag_mode_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "name": "onehot",
                "cache_dir": tmp_dir,
                "cache_enabled": False,
                "onehot_mode": "bag",
            }
            encoder = build_encoder(config)
            features = encoder.encode(["ACDX"])
            self.assertEqual(features.shape, (1, 21))
            self.assertAlmostEqual(float(features.sum()), 1.0, places=5)

    def test_cache_reload_hits_for_onehot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "name": "onehot",
                "cache_dir": tmp_dir,
                "cache_enabled": True,
                "batch_size": 4,
                "onehot_mode": "flattened",
                "fixed_length": 4,
            }
            first = build_encoder(config)
            expected = first.encode(["ACDX", "AAAA"])
            second = build_encoder(config)
            observed = second.encode(["ACDX", "AAAA"])
            self.assertTrue(np.allclose(expected, observed))
            self.assertGreaterEqual(second.get_stats()["cache_hits"], 2)

    def test_variable_length_onehot_cache_namespace_stays_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "name": "onehot",
                "cache_dir": tmp_dir,
                "cache_enabled": True,
                "batch_size": 4,
                "onehot_mode": "flattened",
                "fixed_length": None,
            }
            first = build_encoder(config)
            expected = first.encode(["ACD", "AAA"])
            second = build_encoder(config)
            observed = second.encode(["ACD", "AAA"])
            self.assertTrue(np.allclose(expected, observed))
            self.assertGreaterEqual(second.get_stats()["cache_hits"], 2)
            self.assertEqual(first.get_stats()["feature_dim"], 63)
            self.assertEqual(second.get_stats()["feature_dim"], 63)

    def test_esm_stub_dimension_and_finiteness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "name": "esm",
                "cache_dir": tmp_dir,
                "cache_enabled": True,
                "batch_size": 4,
                "esm_backend": "stub",
                "esm_embedding_dim": 32,
                "allow_stub_fallback": True,
            }
            encoder = build_encoder(config)
            features = encoder.encode(["ACDX", "AAAA"])
            self.assertEqual(features.shape, (2, 32))
            self.assertFalse(np.isnan(features).any())


if __name__ == "__main__":
    unittest.main()
