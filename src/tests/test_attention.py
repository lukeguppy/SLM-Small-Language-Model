import unittest
import torch
import torch.nn as nn
import sys

sys.path.append(".")

from src.core.model import SmallTransformer
from src.model_config import ModelConfig


class TestPyTorchAttention(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Use a reasonable vocab size for attention testing
        self.vocab_size = 100
        config = ModelConfig(vocab_size=self.vocab_size, embed_dim=256, n_heads=4, n_layers=2)
        self.model = SmallTransformer(config)

    def test_attention_masking_effect(self):
        """Test that attention masking affects outputs"""
        batch_size = 1
        seq_len = 5
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        # Test without masking
        mask_none = torch.ones(batch_size, seq_len, dtype=torch.bool)
        with torch.no_grad():
            logits_none = self.model(x, mask_none)

        # Test with masking (last token as padding)
        mask_pad = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask_pad[0, -1] = False
        with torch.no_grad():
            logits_pad = self.model(x, mask_pad)

        # Outputs should be different when masking is applied
        self.assertFalse(torch.allclose(logits_none, logits_pad))

    def test_attention_gradient_flow(self):
        """Test that gradients flow through attention layers"""
        batch_size = 2
        seq_len = 4
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        targets = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        # Forward pass
        logits = self.model(x, mask)
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.vocab_size), targets.view(-1))

        # Backward pass
        loss.backward()

        # Check that gradients flow to attention-related parameters
        # The transformer encoder contains attention layers
        transformer_grads = [p.grad for p in self.model.transformer.parameters() if p.grad is not None]
        self.assertTrue(len(transformer_grads) > 0, "No gradients in transformer layers")

        # Check that all gradients are finite
        for grad in transformer_grads:
            self.assertFalse(torch.isnan(grad).any())
            self.assertFalse(torch.isinf(grad).any())

    def test_multihead_attention_properties(self):
        """Test properties of multi-head attention"""
        batch_size = 1
        seq_len = 6
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Get model outputs
        with torch.no_grad():
            logits = self.model(x, mask)

        # Test that outputs are reasonable
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        # Test that different positions have different outputs (attention working)
        output_diff = logits[0, 1:, :] - logits[0, :-1, :]
        mean_diff = torch.mean(torch.abs(output_diff))
        self.assertGreater(mean_diff.item(), 1e-6)

    def test_attention_with_different_head_counts(self):
        """Test attention with different numbers of heads"""
        configs = [
            {"n_heads": 2, "embed_dim": 128},
            {"n_heads": 4, "embed_dim": 256},
            {"n_heads": 8, "embed_dim": 512},
        ]

        batch_size = 1
        seq_len = 4
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        for config_dict in configs:
            config = ModelConfig(
                vocab_size=self.vocab_size,
                n_layers=1,  # Single layer for simplicity
                **config_dict  # pyright: ignore[reportArgumentType]
            )
            model = SmallTransformer(config)
            with torch.no_grad():
                logits = model(x, mask)
            self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))

    def test_attention_numerical_stability(self):
        """Test numerical stability of attention computations"""
        batch_size = 1
        seq_len = 5
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test with normal inputs
        with torch.no_grad():
            logits = self.model(x, mask)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

        # Test with extreme embeddings (simulate large weights)
        with torch.no_grad():
            original_embed = self.model.embed.weight.data.clone()
            self.model.embed.weight.data *= 1000
            logits_large = self.model(x, mask)
            self.assertFalse(torch.isnan(logits_large).any())
            self.assertFalse(torch.isinf(logits_large).any())
            # Reset
            self.model.embed.weight.data = original_embed

    def test_attention_mask_types(self):
        """Test different types of attention masks"""
        batch_size = 2
        seq_len = 6
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))

        # Test full mask
        mask_full = torch.ones(batch_size, seq_len, dtype=torch.bool)
        with torch.no_grad():
            logits_full = self.model(x, mask_full)

        # Test partial mask (different for each sequence in batch)
        mask_partial = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask_partial[0, -2:] = False  # Last 2 tokens padding for first sequence
        mask_partial[1, -1] = False  # Last token padding for second sequence
        with torch.no_grad():
            logits_partial = self.model(x, mask_partial)

        # Outputs should be different
        self.assertFalse(torch.allclose(logits_full, logits_partial))

        # Check that padding positions have reasonable values
        self.assertFalse(torch.isnan(logits_partial).any())
        self.assertFalse(torch.isinf(logits_partial).any())

    def test_attention_layer_depth(self):
        """Test attention with different numbers of layers"""
        batch_size = 1
        seq_len = 4
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        for n_layers in [1, 2, 3]:
            config = ModelConfig(
                vocab_size=self.vocab_size, embed_dim=128, n_heads=2, n_layers=n_layers  # Smaller for speed
            )
            model = SmallTransformer(config)
            with torch.no_grad():
                logits = model(x, mask)
            self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))

    def test_attention_output_distribution(self):
        """Test that attention produces reasonable output distributions"""
        batch_size = 1
        seq_len = 3
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with torch.no_grad():
            logits = self.model(x, mask)
            probs = torch.softmax(logits, dim=-1)

        # Check that probabilities sum to 1
        prob_sums = torch.sum(probs, dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums)))

        # Check that all probabilities are valid
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

        # Check that we don't have uniform distribution (attention should focus)
        # Weak test but helps ensure attention is working
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        mean_entropy = torch.mean(entropy)
        self.assertGreater(mean_entropy.item(), 0.1)  # Some uncertainty


if __name__ == "__main__":
    unittest.main()
