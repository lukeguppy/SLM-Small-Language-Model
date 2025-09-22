import unittest
import torch
import torch.nn as nn

# Add src to path for imports
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.custom_attention import CustomMultiHeadAttention, CustomTransformerEncoderLayer, CustomTransformerEncoder


class TestCustomMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.embed_dim = 64
        self.num_heads = 4
        self.head_dim = self.embed_dim // self.num_heads

        # Create attention layer
        self.attention = CustomMultiHeadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0)

    def test_initialisation(self):
        """Test proper initialisation of attention layer"""
        self.assertEqual(self.attention.embed_dim, self.embed_dim)
        self.assertEqual(self.attention.num_heads, self.num_heads)
        self.assertEqual(self.attention.head_dim, self.head_dim)

        # Check that projections are created
        self.assertIsInstance(self.attention.q_proj, nn.Linear)
        self.assertIsInstance(self.attention.k_proj, nn.Linear)
        self.assertIsInstance(self.attention.v_proj, nn.Linear)
        self.assertIsInstance(self.attention.out_proj, nn.Linear)

    def test_forward_without_mask(self):
        """Test forward pass without attention mask"""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        output, attention_weights = self.attention(query, key, value)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

        # Check attention weights shape
        expected_weights_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(attention_weights.shape, expected_weights_shape)

        # Check that attention weights sum to 1 along last dimension
        attention_sums = attention_weights.sum(dim=-1)
        torch.testing.assert_close(attention_sums, torch.ones_like(attention_sums), rtol=1e-5, atol=1e-5)

    def test_forward_with_mask(self):
        """Test forward pass with attention mask"""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        # Create causal mask (upper triangular)
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool()

        output, attention_weights = self.attention(query, key, value, mask=mask)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

        # Check that masked positions have zero attention
        # For causal mask, positions above diagonal should be masked
        masked_weights = attention_weights[:, :, mask]
        torch.testing.assert_close(masked_weights, torch.zeros_like(masked_weights), rtol=1e-6, atol=1e-6)

    def test_attention_weights_storage(self):
        """Test that attention weights are properly stored"""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        # Initially should be None
        self.assertIsNone(self.attention.get_attention_weights())

        # After forward pass, should have weights
        self.attention(query, key, value)
        weights = self.attention.get_attention_weights()
        self.assertIsNotNone(weights)

        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(weights.shape, expected_shape)

    def test_need_weights_false(self):
        """Test forward pass when need_weights=False"""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        output, attention_weights = self.attention(query, key, value, need_weights=False)

        # Output should still be correct
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

        # Attention weights should be None
        self.assertIsNone(attention_weights)

    def test_different_sequence_lengths(self):
        """Test with different query/key lengths"""
        query_len = 3
        key_len = 4

        query = torch.randn(self.batch_size, query_len, self.embed_dim)
        key = torch.randn(self.batch_size, key_len, self.embed_dim)
        value = torch.randn(self.batch_size, key_len, self.embed_dim)

        output, attention_weights = self.attention(query, key, value)

        # Check shapes
        self.assertEqual(output.shape, (self.batch_size, query_len, self.embed_dim))
        expected_weights_shape = (self.batch_size, self.num_heads, query_len, key_len)
        self.assertEqual(attention_weights.shape, expected_weights_shape)

    def test_gradient_flow(self):
        """Test that gradients flow through the attention layer"""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim, requires_grad=True)

        output, _ = self.attention(query, key, value)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)


class TestCustomTransformerEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.embed_dim = 64
        self.num_heads = 4
        self.ff_dim = 128

        self.encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=0.0
        )

    def test_initialisation(self):
        """Test proper initialisation of encoder layer"""
        self.assertIsInstance(self.encoder_layer.self_attn, CustomMultiHeadAttention)
        self.assertIsInstance(self.encoder_layer.linear1, nn.Linear)
        self.assertIsInstance(self.encoder_layer.linear2, nn.Linear)
        self.assertIsInstance(self.encoder_layer.norm1, nn.LayerNorm)
        self.assertIsInstance(self.encoder_layer.norm2, nn.LayerNorm)

    def test_forward_pass(self):
        """Test forward pass of encoder layer"""
        src = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        output = self.encoder_layer(src)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

        # Check that attention weights are captured
        weights = self.encoder_layer.get_attention_weights()
        self.assertIsNotNone(weights)
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(weights.shape, expected_shape)

    def test_forward_with_mask(self):
        """Test forward pass with source mask"""
        src = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        # Create mask (e.g., for padding)
        mask = torch.zeros(self.seq_len, self.seq_len).bool()
        mask[2:, 2:] = True  # Mask some positions

        output = self.encoder_layer(src, src_mask=mask)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_residual_connections(self):
        """Test that residual connections are working"""
        src = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        # Store original input
        original_src = src.clone()

        output = self.encoder_layer(src)

        # Output should be different from input (due to residual + layer norm)
        self.assertFalse(torch.equal(output, original_src))


class TestCustomTransformerEncoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.embed_dim = 64
        self.num_heads = 4
        self.ff_dim = 128
        self.num_layers = 3

        # Create base encoder layer
        self.encoder = CustomTransformerEncoder(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.ff_dim,
            dropout=0.0,
        )

    def test_initialisation(self):
        """Test proper initialisation of multi-layer encoder"""
        self.assertEqual(len(self.encoder.layers), self.num_layers)
        self.assertEqual(self.encoder.num_layers, self.num_layers)

        # Check that all layers are CustomTransformerEncoderLayer instances
        for layer in self.encoder.layers:
            self.assertIsInstance(layer, CustomTransformerEncoderLayer)

    def test_forward_pass(self):
        """Test forward pass through multiple layers"""
        src = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        output = self.encoder(src)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_get_attention_weights_single_layer(self):
        """Test getting attention weights from single layer"""
        src = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        # Forward pass to generate weights
        self.encoder(src)

        # Get weights from layer 0
        weights = self.encoder.get_attention_weights(layer_idx=0)
        self.assertIsNotNone(weights)
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(weights.shape, expected_shape)

    def test_get_attention_weights_all_layers(self):
        """Test getting attention weights from all layers"""
        src = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        # Forward pass to generate weights
        self.encoder(src)

        # Get weights from all layers
        weights = self.encoder.get_attention_weights()
        self.assertIsNotNone(weights)
        self.assertIsInstance(weights, list)
        self.assertEqual(len(weights), self.num_layers)

        # Check shape of each layer's weights
        for layer_weights in weights:
            expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
            self.assertEqual(layer_weights.shape, expected_shape)

    def test_get_attention_weights_invalid_layer(self):
        """Test getting attention weights from invalid layer index"""
        src = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.encoder(src)

        # Invalid layer index
        weights = self.encoder.get_attention_weights(layer_idx=10)
        self.assertIsNone(weights)

    def test_layer_independence(self):
        """Test that layers maintain independent attention weights"""
        src = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

        self.encoder(src)

        # Get weights from different layers
        weights_0 = self.encoder.get_attention_weights(layer_idx=0)
        weights_1 = self.encoder.get_attention_weights(layer_idx=1)

        # Weights should be different (layers process sequentially)
        self.assertFalse(torch.equal(weights_0, weights_1))


if __name__ == "__main__":
    unittest.main()
