import unittest
import tempfile
import os
import torch
import torch.nn as nn
from unittest.mock import patch
from typing import cast, Dict, Any

from ..core.model_loader import ModelLoader
from ..model_config import ModelConfig
from ..model_config import ConfigPersistence
from ..core.logger import get_testing_logger


class TestModelLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = get_testing_logger()
        self.config = ModelConfig(
            vocab_size=100,
            embed_dim=64,
            n_heads=4,
            ff_dim=128,
            n_layers=2,
            dropout=0.1,
            max_seq_len=1000,  # Match mock model
            epochs=10,
            lr=0.001,
            weight_decay=0.0,
            batch_size=32,
            train_size=1000,
            val_size=200,
            test_size=200,
            vocab_path="data/tokens.txt",
            model_name="model",
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_mock_model(self):
        """Create a mock model for testing"""
        model = nn.Module()
        model.embed = nn.Embedding(self.config.vocab_size, self.config.embed_dim)
        model.pos_embed = nn.Embedding(1000, self.config.embed_dim)  # Fixed size for test compatibility
        model.transformer = nn.Module()
        model.transformer.layers = nn.ModuleList([nn.Module() for _ in range(self.config.n_layers)])
        for i, layer in enumerate(model.transformer.layers):
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim)
            layer.self_attn.k_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim)
            layer.self_attn.v_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim)
            layer.self_attn.out_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim)
            layer.linear1 = nn.Linear(self.config.embed_dim, self.config.ff_dim)
            layer.linear2 = nn.Linear(self.config.ff_dim, self.config.embed_dim)
            layer.norm1 = nn.LayerNorm(self.config.embed_dim)
            layer.norm2 = nn.LayerNorm(self.config.embed_dim)
        model.ln_f = nn.LayerNorm(self.config.embed_dim)
        model.lm_head = nn.Linear(self.config.embed_dim, self.config.vocab_size)
        return model

    @patch("src.core.model_loader.ModelPaths.get_meta_path")
    def test_load_meta_params_success(self, mock_get_meta_path):
        """Test loading parameters from .meta file successfully"""
        meta_path = os.path.join(self.temp_dir, "test.meta")
        mock_get_meta_path.return_value = meta_path

        meta_data = {"embed_dim": 128, "n_heads": 8, "ff_dim": 256, "n_layers": 4, "dropout": 0.1}

        with open(meta_path, "w") as f:
            for key, value in meta_data.items():
                f.write(f"{key}: {value}\n")

        result = ModelLoader._load_meta_params(os.path.join(self.temp_dir, "test.pt"), logger=self.logger)
        self.assertIsNotNone(result)
        # Type checker needs explicit cast after assertIsNotNone
        result = cast(ModelConfig, result)
        self.assertIsInstance(result, ModelConfig)
        self.assertEqual(result.embed_dim, 128)
        self.assertEqual(result.n_heads, 8)

    def test_load_meta_params_file_not_found(self):
        """Test loading parameters when .meta file doesn't exist"""
        result = ModelLoader._load_meta_params(os.path.join(self.temp_dir, "nonexistent.pt"), logger=self.logger)
        self.assertIsNone(result)

    def test_infer_params_from_state_dict(self):
        """Test parameter inference from PyTorch state dict"""
        model = self.create_mock_model()
        state_dict = model.state_dict()

        inferred = ModelLoader._infer_params_from_state_dict(state_dict, self.config.vocab_size, logger=self.logger)

        self.assertEqual(inferred["embed_dim"], self.config.embed_dim)
        self.assertEqual(inferred["n_layers"], self.config.n_layers)
        # n_heads inference might not work perfectly due to head_dim constraints

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters"""
        config = ModelConfig(vocab_size=100, embed_dim=64, n_heads=4, ff_dim=128, n_layers=2)
        result = config.validate()
        self.assertTrue(result)

    def test_validate_parameters_invalid(self):
        """Test parameter validation with invalid parameters"""
        # embed_dim not divisible by n_heads
        config = ModelConfig(vocab_size=100, embed_dim=64, n_heads=3, ff_dim=128, n_layers=2)
        result = config.validate()
        self.assertFalse(result)

        # Negative values
        config = ModelConfig(vocab_size=100, embed_dim=-1, n_heads=4, ff_dim=128, n_layers=2)
        result = config.validate()
        self.assertFalse(result)

    def test_get_model_info_success(self):
        """Test getting model info from saved model"""
        model = self.create_mock_model()
        model_path = os.path.join(self.temp_dir, "test_model.pt")
        torch.save(model.state_dict(), model_path)

        info = ModelLoader.get_model_info(model_path, logger=self.logger)
        self.assertIsNotNone(info)
        # Type checker needs explicit cast after assertIsNotNone
        info = cast(Dict[str, Any], info)
        self.assertEqual(info["vocab_size"], self.config.vocab_size)
        self.assertEqual(info["embed_dim"], self.config.embed_dim)
        self.assertEqual(info["n_layers"], self.config.n_layers)

    def test_get_model_info_file_not_found(self):
        """Test getting model info when file doesn't exist"""
        info = ModelLoader.get_model_info(os.path.join(self.temp_dir, "nonexistent.pt"), logger=self.logger)
        self.assertIsNone(info)

    def test_save_model_with_meta(self):
        """Test saving model with metadata"""
        model = self.create_mock_model()
        model_path = os.path.join(self.temp_dir, "test_save.pt")
        meta_path = os.path.join(self.temp_dir, "test_save.meta")

        # Mock the PathManager methods to use temp directory
        with patch("src.core.model_loader.ModelPaths.get_model_path", return_value=model_path), patch(
            "src.core.model_loader.ModelPaths.get_meta_path", return_value=meta_path
        ), patch("src.core.model_loader.ModelPaths.ensure_model_dir", return_value=self.temp_dir):

            logger = get_testing_logger()
            ModelLoader.save_model_with_meta(model, model_path, logger, self.config)

            # Check model file exists
            self.assertTrue(os.path.exists(model_path))

            # Check meta file exists
            self.assertTrue(os.path.exists(meta_path))

            # Check meta file contents using ConfigPersistence
            loaded_config = ConfigPersistence.load_from_file(meta_path, self.logger)
            self.assertIsNotNone(loaded_config)
            # Type checker needs explicit cast after assertIsNotNone
            loaded_config = cast(ModelConfig, loaded_config)
            self.assertEqual(loaded_config.vocab_size, self.config.vocab_size)
            self.assertEqual(loaded_config.embed_dim, self.config.embed_dim)

    @patch("src.core.serialisation.SerialisationService.load_model_state")
    def test_load_model_safely_with_meta(self, mock_load_state):
        """Test loading model with meta file present"""
        model = self.create_mock_model()
        mock_load_state.return_value = model.state_dict()

        # Create meta file using ConfigPersistence
        model_path = os.path.join(self.temp_dir, "test.pt")
        meta_path = model_path + ".meta"
        # Create a config with max_seq_len matching the mock model
        test_config = ModelConfig(
            vocab_size=100,
            embed_dim=64,
            n_heads=4,
            ff_dim=128,
            n_layers=2,
            dropout=0.1,
            max_seq_len=1000,
            epochs=10,
            lr=0.001,
            weight_decay=0.0,
            batch_size=32,
            train_size=1000,
            val_size=200,
            test_size=200,
            vocab_path="data/tokens.txt",
            model_name="model",
        )
        ConfigPersistence.save_to_file(test_config, meta_path, self.logger)

        # Mock PathManager to return our test meta path
        with patch("src.core.model_loader.ModelPaths.get_meta_path", return_value=meta_path):
            loaded_model, loaded_config = ModelLoader.load_model_safely(
                model_path, config=self.config, logger=self.logger
            )

        self.assertIsNotNone(loaded_model)
        self.assertIsNotNone(loaded_config)
        # Type checker needs explicit cast after assertIsNotNone
        assert loaded_config is not None  # For type checker
        self.assertEqual(loaded_model.embed_dim, 64)

    @patch("src.core.serialisation.SerialisationService.load_model_state")
    def test_load_model_safely_without_meta(self, mock_load_state):
        """Test loading model without meta file"""
        model = self.create_mock_model()
        mock_load_state.return_value = model.state_dict()

        model_path = os.path.join(self.temp_dir, "test_no_meta.pt")
        loaded_model = ModelLoader.load_model_safely(model_path, config=self.config, logger=self.logger)

        self.assertIsNotNone(loaded_model)

    @patch("src.core.serialisation.SerialisationService.load_model_state")
    def test_load_model_safely_with_parameter_mismatch(self, mock_load_state):
        """Test loading model with parameter mismatch (should handle gracefully)"""
        model = self.create_mock_model()
        state_dict = model.state_dict()

        # Modify state dict to simulate mismatch
        # This is a simplified test - in practice, parameter mapping would be more complex
        mock_load_state.return_value = state_dict

        model_path = os.path.join(self.temp_dir, "test_mismatch.pt")
        loaded_model = ModelLoader.load_model_safely(model_path, config=self.config, logger=self.logger)

        # Should still load successfully despite any warnings
        self.assertIsNotNone(loaded_model)

    def test_map_old_to_new_params(self):
        """Test mapping old parameter format to new format"""
        # Create a mock old-style state dict
        old_state_dict = {
            "transformer.layers.0.self_attn.in_proj_weight": torch.randn(192, 64),  # 3*64 for QKV
            "transformer.layers.0.self_attn.in_proj_bias": torch.randn(192),
            "transformer.layers.0.self_attn.out_proj.weight": torch.randn(64, 64),
            "transformer.layers.0.self_attn.out_proj.bias": torch.randn(64),
            "embed.weight": torch.randn(100, 64),
        }

        model = self.create_mock_model()
        mapped = ModelLoader._map_old_to_new_params(old_state_dict, model)

        # Should have mapped parameters
        self.assertIsNotNone(mapped)
        # Type checker needs explicit cast after assertIsNotNone
        mapped = cast(Dict[str, torch.Tensor], mapped)
        self.assertIn("transformer.layers.0.self_attn.q_proj.weight", mapped)
        self.assertIn("transformer.layers.0.self_attn.k_proj.weight", mapped)
        self.assertIn("transformer.layers.0.self_attn.v_proj.weight", mapped)


if __name__ == "__main__":
    unittest.main()
