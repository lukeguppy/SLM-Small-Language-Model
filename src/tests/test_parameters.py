import unittest
import tempfile
import os
import torch
import torch.nn as nn
from typing import cast

from ..model_config import ModelConfig
from ..core.model import SmallTransformer
from ..model_config import ConfigPersistence
from ..core.logger import get_testing_logger


class TestParameters(unittest.TestCase):
    """Test class for parameter validation"""

    def setUp(self):
        self.vocab_size = 1000  # Test vocab size

    def test_config_parameters(self):
        """Test that all config parameters are properly loaded"""
        config = ModelConfig()

        # Test each parameter
        expected_values = {
            "epochs": 50,  # Updated default
            "lr": 0.0005,  # Updated default
            "train_size": 50000,
            "val_size": 10000,
            "test_size": 10000,
            "embed_dim": 512,
            "ff_dim": 1024,
            "n_heads": 8,
            "n_layers": 4,
            "dropout": 0.3,
            "weight_decay": 0.0001,
            "batch_size": 64,  # Updated default
        }

        for param, expected in expected_values.items():
            actual = getattr(config, param)
            self.assertEqual(actual, expected, f"Config {param}: expected {expected}, got {actual}")

    def test_model_creation_with_parameters(self):
        """Test that model can be created with different parameter combinations"""

        # Test default parameters
        config_default = ModelConfig(vocab_size=self.vocab_size)
        model_default = SmallTransformer(config_default)
        self.assertIsNotNone(model_default)

        # Test custom embed_dim
        config_embed = ModelConfig(vocab_size=self.vocab_size, embed_dim=256)
        model_embed = SmallTransformer(config_embed)
        self.assertEqual(model_embed.embed.embedding_dim, 256)

        # Test custom ff_dim
        config_ff = ModelConfig(vocab_size=self.vocab_size, ff_dim=512)
        model_ff = SmallTransformer(config_ff)
        # Type assertion for type checker
        linear1 = cast(nn.Linear, model_ff.transformer.layers[0].linear1)
        self.assertEqual(cast(nn.Linear, linear1).out_features, 512)

        # Test custom n_heads
        config_heads = ModelConfig(vocab_size=self.vocab_size, n_heads=4)
        model_heads = SmallTransformer(config_heads)
        # Note: Can't easily check n_heads from model, but creation should succeed
        self.assertIsNotNone(model_heads)

        # Test custom n_layers
        config_layers = ModelConfig(vocab_size=self.vocab_size, n_layers=2)
        model_layers = SmallTransformer(config_layers)
        self.assertEqual(len(model_layers.transformer.layers), 2)

        # Test custom dropout
        config_dropout = ModelConfig(vocab_size=self.vocab_size, dropout=0.1)
        model_dropout = SmallTransformer(config_dropout)
        # Check dropout in first layer
        dropout1 = cast(nn.Dropout, model_dropout.transformer.layers[0].dropout1)
        self.assertEqual(cast(nn.Dropout, dropout1).p, 0.1)

    def test_parameter_ranges(self):
        """Test parameter validation and edge cases"""

        # Test embed_dim range
        with self.assertRaises((ValueError, AssertionError)):
            config = ModelConfig(vocab_size=self.vocab_size, embed_dim=0)
            SmallTransformer(config)

        # Test ff_dim range
        with self.assertRaises((ValueError, AssertionError)):
            config = ModelConfig(vocab_size=self.vocab_size, ff_dim=0)
            SmallTransformer(config)

        # Test n_heads range
        with self.assertRaises((ValueError, AssertionError)):
            config = ModelConfig(vocab_size=self.vocab_size, n_heads=0)
            SmallTransformer(config)

        # Test n_layers range
        with self.assertRaises((ValueError, AssertionError)):
            config = ModelConfig(vocab_size=self.vocab_size, n_layers=0)
            SmallTransformer(config)

        # Test dropout range - PyTorch Dropout validates p must be between 0 and 1
        with self.assertRaises(ValueError):
            config_neg = ModelConfig(vocab_size=self.vocab_size, dropout=-0.1)
            SmallTransformer(config_neg)

        with self.assertRaises(ValueError):
            config_high = ModelConfig(vocab_size=self.vocab_size, dropout=1.5)
            SmallTransformer(config_high)

    def test_model_forward_with_parameters(self):
        """Test that model forward pass works with various parameter combinations"""
        batch_size = 2
        seq_len = 5

        # Test different embed_dims
        for embed_dim in [128, 256, 512]:
            config = ModelConfig(vocab_size=self.vocab_size, embed_dim=embed_dim)
            model = SmallTransformer(config)
            x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

            logits = model(x, mask)
            expected_shape = (batch_size, seq_len, self.vocab_size)
            self.assertEqual(logits.shape, expected_shape, f"Wrong shape for embed_dim={embed_dim}: {logits.shape}")

        # Test different ff_dims
        for ff_dim in [256, 512, 1024]:
            config = ModelConfig(vocab_size=self.vocab_size, ff_dim=ff_dim)
            model = SmallTransformer(config)
            x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

            logits = model(x, mask)
            expected_shape = (batch_size, seq_len, self.vocab_size)
            self.assertEqual(logits.shape, expected_shape, f"Wrong shape for ff_dim={ff_dim}: {logits.shape}")

        # Test different n_layers
        for n_layers in [1, 2, 4]:
            config = ModelConfig(vocab_size=self.vocab_size, n_layers=n_layers)
            model = SmallTransformer(config)
            x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

            logits = model(x, mask)
            expected_shape = (batch_size, seq_len, self.vocab_size)
            self.assertEqual(logits.shape, expected_shape, f"Wrong shape for n_layers={n_layers}: {logits.shape}")

    def test_config_modification(self):
        """Test that config can be modified and affects model creation"""
        config = ModelConfig()

        # Modify config values
        config.embed_dim = 256
        config.ff_dim = 512
        config.n_heads = 4
        config.n_layers = 2
        config.dropout = 0.1

        # Create model with modified config
        model = SmallTransformer(config)

        # Verify parameters
        self.assertEqual(model.embed.embedding_dim, config.embed_dim)
        # Type assertions for type checker
        linear1 = cast(nn.Linear, model.transformer.layers[0].linear1)
        self.assertEqual(cast(nn.Linear, linear1).out_features, config.ff_dim)
        self.assertEqual(len(model.transformer.layers), config.n_layers)
        dropout1 = cast(nn.Dropout, model.transformer.layers[0].dropout1)
        self.assertEqual(cast(nn.Dropout, dropout1).p, config.dropout)

    def test_meta_file_save_load(self):
        """Test that .meta files properly save and load ff_dim using ConfigPersistence"""
        # Create a test config and model
        test_config = ModelConfig(
            vocab_size=self.vocab_size,
            ff_dim=1024,
            embed_dim=256,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            lr=0.001,
            epochs=10,
            weight_decay=0.0001,
            batch_size=16,
            train_size=1000,
            val_size=200,
            test_size=200,
            vocab_path="data/tokens.txt",
        )
        test_model = SmallTransformer(test_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save config using ConfigPersistence.save_to_file
            meta_path = os.path.join(temp_dir, "test_model.meta")
            logger = get_testing_logger()
            ConfigPersistence.save_to_file(test_config, meta_path, logger)

            # Load config using ConfigPersistence.load_from_file
            loaded_config = ConfigPersistence.load_from_file(meta_path, logger)
            self.assertIsNotNone(loaded_config, "Failed to load config from file")

            # Verify ff_dim is saved and loaded correctly
            self.assertEqual(loaded_config.ff_dim, 1024, f"ff_dim loaded incorrectly: {loaded_config.ff_dim}")

            # Test that model can be created with loaded config
            loaded_model = SmallTransformer(loaded_config)

            # Verify the loaded model has correct ff_dim
            loaded_linear1 = cast(nn.Linear, loaded_model.transformer.layers[0].linear1)
            self.assertEqual(cast(nn.Linear, loaded_linear1).out_features, 1024)


if __name__ == "__main__":
    unittest.main()
