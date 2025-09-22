import unittest
import torch
import torch.nn as nn
import tempfile
import os
from typing import cast

from ..core.model import SmallTransformer
from ..model_config import ModelConfig
from ..core.custom_attention import CustomTransformerEncoder


class TestSmallTransformer(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(vocab_size=100, embed_dim=64, n_heads=4, ff_dim=128, n_layers=2, dropout=0.1)

        self.model = SmallTransformer(self.config)

    def test_initialisation(self):
        """Test proper initialisation of SmallTransformer"""
        self.assertEqual(self.model.vocab_size, self.config.vocab_size)
        self.assertEqual(self.model.embed_dim, self.config.embed_dim)
        self.assertEqual(self.model.n_heads, self.config.n_heads)
        self.assertEqual(self.model.n_layers, self.config.n_layers)

        # Check that components are created
        self.assertIsInstance(self.model.embed, nn.Embedding)
        self.assertIsInstance(self.model.pos_embed, nn.Embedding)
        self.assertIsInstance(self.model.transformer, CustomTransformerEncoder)
        self.assertIsInstance(self.model.fc_out, nn.Linear)

    def test_forward_pass(self):
        """Test forward pass with valid inputs"""
        batch_size = 2
        seq_len = 5

        # Create input tensors
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = self.model(x, mask)

        # Check output shape
        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_pass_different_lengths(self):
        """Test forward pass with different sequence lengths"""
        batch_size = 3
        seq_len = 7

        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = self.model(x, mask)

        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_pass_with_padding(self):
        """Test forward pass with padding mask"""
        batch_size = 2
        seq_len = 6

        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        # Create mask with some padding (False for padding positions)
        mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, False, False]])

        output = self.model(x, mask)

        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(output.shape, expected_shape)

    def test_parameter_validation(self):
        """Test that parameters are validated correctly"""
        # Valid parameters should work
        config = ModelConfig(vocab_size=50, embed_dim=32, n_heads=2, ff_dim=64, n_layers=1)
        model = SmallTransformer(config)
        self.assertIsInstance(model, SmallTransformer)

        # embed_dim should be divisible by n_heads - test validation method
        config_invalid = ModelConfig(vocab_size=50, embed_dim=32, n_heads=3, ff_dim=64, n_layers=1)
        self.assertFalse(
            config_invalid.validate(), "Config validation should fail when embed_dim not divisible by n_heads"
        )

    def test_output_logits(self):
        """Test that output logits have correct properties"""
        batch_size = 1
        seq_len = 3

        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = self.model(x, mask)

        # Check that we have logits for each position and vocab item
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.vocab_size))

        # Check that logits are reasonable (not all same value)
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))

    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        batch_size = 2
        seq_len = 4

        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = self.model(x, mask)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for embeddings
        self.assertIsNotNone(self.model.embed.weight.grad)

    def test_model_save_load(self):
        """Test saving and loading model state"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pt")

            # Save model
            torch.save(self.model.state_dict(), model_path)

            # Create new model and load
            new_model = SmallTransformer(self.config)
            new_model.load_state_dict(torch.load(model_path))

            # Test that loaded model has same structure
            self.assertEqual(new_model.vocab_size, self.model.vocab_size)
            self.assertEqual(new_model.embed_dim, self.model.embed_dim)
            self.assertEqual(new_model.n_heads, self.model.n_heads)
            self.assertEqual(new_model.n_layers, self.model.n_layers)

            # Test that loaded model can forward pass
            x = torch.randint(0, self.config.vocab_size, (1, 3))
            mask = torch.ones(1, 3, dtype=torch.bool)

            loaded_output = new_model(x, mask)
            self.assertEqual(loaded_output.shape, (1, 3, self.config.vocab_size))

    def test_different_vocab_sizes(self):
        """Test model with different vocabulary sizes"""
        small_config = ModelConfig(vocab_size=10, embed_dim=32, n_heads=2, ff_dim=64, n_layers=1)
        large_config = ModelConfig(vocab_size=1000, embed_dim=32, n_heads=2, ff_dim=64, n_layers=1)
        small_vocab_model = SmallTransformer(small_config)
        large_vocab_model = SmallTransformer(large_config)

        # Both should work
        x_small = torch.randint(0, 10, (1, 2))
        x_large = torch.randint(0, 1000, (1, 2))
        mask = torch.ones(1, 2, dtype=torch.bool)

        output_small = small_vocab_model(x_small, mask)
        output_large = large_vocab_model(x_large, mask)

        self.assertEqual(output_small.shape, (1, 2, 10))
        self.assertEqual(output_large.shape, (1, 2, 1000))

    def test_attention_mask_application(self):
        """Test that attention masks are properly applied"""
        batch_size = 1
        seq_len = 4

        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        # Mask that blocks attention to the last position
        mask = torch.tensor([[True, True, True, False]])

        output = self.model(x, mask)

        # Output should still have correct shape
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.vocab_size))

        # The masked position should still produce output (just not attend to itself)
        self.assertFalse(torch.isnan(output).any())

    def test_model_components(self):
        """Test that all model components are properly connected"""
        # Check embedding layer
        self.assertIsInstance(self.model.embed, nn.Embedding)
        # Type checker needs explicit cast after assertIsInstance
        embed_layer = cast(nn.Embedding, self.model.embed)
        self.assertEqual(embed_layer.num_embeddings, self.config.vocab_size)
        self.assertEqual(embed_layer.embedding_dim, self.config.embed_dim)

        # Check output layer
        self.assertIsInstance(self.model.fc_out, nn.Linear)
        # Type checker needs explicit cast after assertIsInstance
        fc_out_layer = cast(nn.Linear, self.model.fc_out)
        self.assertEqual(fc_out_layer.in_features, self.config.embed_dim)
        self.assertEqual(fc_out_layer.out_features, self.config.vocab_size)

        # Check transformer layers
        self.assertEqual(len(self.model.transformer.layers), self.config.n_layers)


if __name__ == "__main__":
    unittest.main()
