import unittest
from ..model_config import ModelConfig


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig class"""

    def test_default_initialisation(self):
        """Test default initialisation values"""
        config = ModelConfig()

        # Check default values
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.embed_dim, 512)
        self.assertEqual(config.ff_dim, 1024)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.dropout, 0.3)
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.lr, 0.0005)
        self.assertEqual(config.weight_decay, 0.0001)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.train_size, 50000)
        self.assertEqual(config.val_size, 10000)
        self.assertEqual(config.test_size, 10000)
        # vocab_path should be set to the full path by __post_init__
        self.assertTrue(config.vocab_path.endswith("data\\tokens.txt") or config.vocab_path.endswith("data/tokens.txt"))
        self.assertEqual(config.max_seq_len, 2048)
        self.assertEqual(config.model_name, "model")

        # Check that saved_date and saved_time are set by __post_init__
        self.assertIsNotNone(config.saved_date)
        self.assertIsNotNone(config.saved_time)
        self.assertIsNone(config.model_path)

    def test_custom_initialisation(self):
        """Test custom initialisation with provided values"""
        config = ModelConfig(
            vocab_size=500,
            embed_dim=256,
            ff_dim=512,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            epochs=100,
            lr=0.001,
            weight_decay=0.01,
            batch_size=32,
            train_size=10000,
            val_size=2000,
            test_size=2000,
            vocab_path="custom/path.txt",
            max_seq_len=1024,
            model_name="custom_model",
            saved_date="2024-01-01",
            saved_time="12:00",
            model_path="/path/to/model",
        )

        # Check custom values
        self.assertEqual(config.vocab_size, 500)
        self.assertEqual(config.embed_dim, 256)
        self.assertEqual(config.ff_dim, 512)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.lr, 0.001)
        self.assertEqual(config.weight_decay, 0.01)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.train_size, 10000)
        self.assertEqual(config.val_size, 2000)
        self.assertEqual(config.test_size, 2000)
        self.assertEqual(config.vocab_path, "custom/path.txt")
        self.assertEqual(config.max_seq_len, 1024)
        self.assertEqual(config.model_name, "custom_model")
        self.assertEqual(config.saved_date, "2024-01-01")
        self.assertEqual(config.saved_time, "12:00")
        self.assertEqual(config.model_path, "/path/to/model")

    def test_post_init_sets_metadata(self):
        """Test that __post_init__ sets saved_date and saved_time when None"""
        # Test with None values (should be set)
        config = ModelConfig(saved_date=None, saved_time=None)
        self.assertIsNotNone(config.saved_date)
        self.assertIsNotNone(config.saved_time)

        # Test with provided values (should not be overwritten)
        config2 = ModelConfig(saved_date="2024-01-01", saved_time="12:00")
        self.assertEqual(config2.saved_date, "2024-01-01")
        self.assertEqual(config2.saved_time, "12:00")

    def test_to_dict(self):
        """Test to_dict method"""
        config = ModelConfig(
            vocab_size=100, embed_dim=64, n_heads=2, n_layers=1, dropout=0.1, train_size=1000, model_name="test_model"
        )

        config_dict = config.to_dict()

        # Check that all expected keys are present
        expected_keys = [
            "vocab_size",
            "embed_dim",
            "n_heads",
            "ff_dim",
            "n_layers",
            "dropout",
            "lr",
            "epochs",
            "weight_decay",
            "batch_size",
            "train_size",
            "val_size",
            "test_size",
            "vocab_path",
            "max_seq_len",
            "model_name",
            "saved_date",
            "saved_time",
        ]

        for key in expected_keys:
            self.assertIn(key, config_dict)

        # Check specific values
        self.assertEqual(config_dict["vocab_size"], 100)
        self.assertEqual(config_dict["embed_dim"], 64)
        self.assertEqual(config_dict["n_heads"], 2)
        self.assertEqual(config_dict["n_layers"], 1)
        self.assertEqual(config_dict["dropout"], 0.1)
        self.assertEqual(config_dict["train_size"], 1000)
        self.assertEqual(config_dict["model_name"], "test_model")

    def test_from_dict(self):
        """Test from_dict method"""
        data = {
            "vocab_size": 200,
            "embed_dim": 128,
            "n_heads": 4,
            "ff_dim": 256,
            "n_layers": 2,
            "dropout": 0.2,
            "lr": 0.001,
            "epochs": 25,
            "weight_decay": 0.001,
            "batch_size": 16,
            "train_size": 5000,
            "val_size": 1000,
            "test_size": 1000,
            "vocab_path": "test/path.txt",
            "max_seq_len": 512,
            "model_name": "from_dict_model",
            "saved_date": "2024-01-01",
            "saved_time": "10:30",
        }

        config = ModelConfig.from_dict(data)

        # Check that values were set correctly
        self.assertEqual(config.vocab_size, 200)
        self.assertEqual(config.embed_dim, 128)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.ff_dim, 256)
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.lr, 0.001)
        self.assertEqual(config.epochs, 25)
        self.assertEqual(config.weight_decay, 0.001)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.train_size, 5000)
        self.assertEqual(config.val_size, 1000)
        self.assertEqual(config.test_size, 1000)
        self.assertEqual(config.vocab_path, "test/path.txt")
        self.assertEqual(config.max_seq_len, 512)
        self.assertEqual(config.model_name, "from_dict_model")
        self.assertEqual(config.saved_date, "2024-01-01")
        self.assertEqual(config.saved_time, "10:30")

    def test_from_dict_ignores_unknown_fields(self):
        """Test from_dict method ignores unknown fields"""
        data = {"vocab_size": 100, "embed_dim": 64, "unknown_field": "should_be_ignored", "another_unknown": 123}

        config = ModelConfig.from_dict(data)

        # Check that known fields were set
        self.assertEqual(config.vocab_size, 100)
        self.assertEqual(config.embed_dim, 64)


if __name__ == "__main__":
    unittest.main()
