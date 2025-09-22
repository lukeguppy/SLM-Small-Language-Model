import unittest
import torch
import torch.nn as nn
import tempfile
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ..training.trainer import Trainer, SentenceDataset, collate_fn
from ..services.vocab_service import VocabService
from ..services.model_service import ModelService
from ..services.data_service import DataService
from ..core.model import SmallTransformer
from ..model_config import ModelConfig
from ..core.file_ops import FileOperations
from ..core.logger import get_testing_logger

# Import for test model creation
from ..core.model import SmallTransformer


class TestPyTorchTrainer(unittest.TestCase):
    def setUp(self):
        # Create test vocabulary
        logger = get_testing_logger()
        self.vocab_service = VocabService(logger=logger)
        self.model_service = ModelService(logger=logger)
        self.data_service = DataService(logger=logger)

        # Create test vocab file
        import tempfile
        import os

        self.temp_dir = tempfile.mkdtemp()
        vocab_content = "hello\nworld\ntest\nsentence\n"
        vocab_path = os.path.join(self.temp_dir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write(vocab_content)

        # Load vocabulary
        self.vocab_service.load_vocabulary(vocab_path)

        # Create model config
        vocab_size = self.vocab_service.get_vocab_size()
        self.config = ModelConfig(vocab_size=vocab_size, embed_dim=128, n_heads=2, n_layers=1)
        self.model = SmallTransformer(self.config)

        # Set model in service
        self.model_service.model = self.model
        self.model_service.config = self.config

        # Create trainer with dependency injection
        self.trainer = Trainer(
            vocab_service=self.vocab_service,
            model_service=self.model_service,
            data_service=self.data_service,
            logger=logger,
        )

    def test_data_loaders_creation(self):
        """Test that data loaders are created properly through trainer"""
        try:
            # Create test data files
            vocab_file = os.path.join(self.temp_dir, "vocab.txt")
            sentences_file = os.path.join(self.temp_dir, "sentences.txt")

            with open(vocab_file, "w") as f:
                f.write("hello\nworld\ntest\nsentence\n")

            with open(sentences_file, "w") as f:
                f.write("hello world\ntest sentence\nworld test\n")

            # Load test data
            self.vocab_service.load_vocabulary(vocab_file)
            self.data_service.load_data(vocab_path=vocab_file)

            # Create config for testing
            config = ModelConfig(
                vocab_size=self.vocab_service.get_vocab_size(), embed_dim=64, n_heads=2, n_layers=1, batch_size=2
            )

            # Test data loader creation through trainer's internal method
            tokenised_sentences = self.data_service.filter_sentences_by_vocab(self.vocab_service.get_vocab())
            if tokenised_sentences:
                data_loaders = self.trainer._create_data_loaders(config, tokenised_sentences)

                # Check that loaders are created
                self.assertIsNotNone(data_loaders)
                self.assertIn("train_loader", data_loaders)
                self.assertIn("val_loader", data_loaders)

                # Check that we can iterate
                train_loader = data_loaders["train_loader"]
                if len(train_loader) > 0:
                    train_batch = next(iter(train_loader))
                    x, y, mask = train_batch

                    self.assertIsInstance(x, torch.Tensor)
                    self.assertIsInstance(y, torch.Tensor)
                    self.assertIsInstance(mask, torch.Tensor)

        except Exception as e:
            if "data files not found" in str(e).lower() or "no valid training sentences" in str(e).lower():
                self.skipTest("Test data setup failed, skipping data loader test")
            else:
                raise

    def test_sentence_dataset(self):
        """Test SentenceDataset functionality"""
        # Create test data
        test_sentences = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        dataset = SentenceDataset(test_sentences)

        self.assertEqual(len(dataset), 3)

        # Testgetitem
        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(len(x), 2)  # input sequence
        self.assertEqual(len(y), 2)  # target sequence

    def test_collate_function(self):
        """Test the collate function for batching"""
        # Create test data - need at least 2 tokens per sentence for valid x,y
        test_sentences = [[1, 2], [4, 5, 6], [8, 9]]
        dataset = SentenceDataset(test_sentences)
        batch = [dataset[i] for i in range(len(dataset))]
        x_batch, y_batch, mask_batch = collate_fn(batch)

        # Check shapes
        batch_size = len(test_sentences)
        max_len = 2  # max input length (from [4,5,6] -> [4,5])
        self.assertEqual(x_batch.shape, (batch_size, max_len))
        self.assertEqual(y_batch.shape, (batch_size, max_len))
        self.assertEqual(mask_batch.shape, (batch_size, max_len))

        # Check masking
        self.assertTrue(mask_batch[0, 0])  # First sequence [1,2] -> x=[1], so position 0 valid
        self.assertFalse(mask_batch[0, 1])  # First sequence, second position (padding)
        self.assertTrue(torch.all(mask_batch[1]))  # Second sequence [4,5,6] -> x=[4,5], all positions valid
        self.assertTrue(mask_batch[2, 0])  # Third sequence [8,9] -> x=[8], first position valid
        self.assertFalse(mask_batch[2, 1])  # Third sequence, second position (padding)

    def test_train_model_function(self):
        """Test the train_model function with temporary files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_file = os.path.join(temp_dir, "tokens.txt")
            sentences_file = os.path.join(temp_dir, "sentences.txt")

            # Create test vocabulary (must include <PAD> handling)
            with open(vocab_file, "w") as f:
                f.write("hello\nworld\ntest\n")

            # Create test sentences (must have at least 2 tokens each for training)
            with open(sentences_file, "w") as f:
                f.write("hello world\ntest hello\nworld test hello\n")

            try:
                # Load the test vocabulary first to get correct size
                self.vocab_service.load_vocabulary(vocab_file)
                actual_vocab_size = self.vocab_service.get_vocab_size()

                # Create a new model with the correct vocabulary size
                test_config = ModelConfig(
                    vocab_size=actual_vocab_size,
                    embed_dim=128,
                    n_heads=2,
                    ff_dim=256,
                    n_layers=1,
                    dropout=0.1,
                    epochs=1,
                    lr=0.01,
                    batch_size=1,
                    train_size=2,  # Small dataset
                    val_size=1,
                    test_size=1,
                    vocab_path=vocab_file,
                )
                test_config.model_name = "test_model"

                # Create new model and set in service
                test_model = SmallTransformer(test_config)
                self.model_service.model = test_model
                self.model_service.config = test_config

                # Test training for 1 epoch with config
                trained_model = self.trainer.train_model(config=test_config)

                # Verify the model was trained and returned
                self.assertIsInstance(trained_model, SmallTransformer)
                self.assertIsNotNone(trained_model)

                # Clean up saved models
                logger = get_testing_logger()
                FileOperations.delete_model_files("test_model", logger)

            except Exception as e:
                # If training fails due to file issues, provide helpful error
                if "data/tokens.txt" in str(e) or "data/sentences.txt" in str(e):
                    self.skipTest("Test data files not found - ensure data/tokens.txt and data/sentences.txt exist")
                else:
                    raise

    def test_generate_text_function(self):
        """Test text generation function"""
        # Test with a simple seed
        vocab = self.vocab_service.get_vocab()
        seed_token = "hello" if "hello" in vocab else list(vocab.keys())[0]

        try:
            # Ensure model is on CPU for testing
            self.model.to("cpu")
            generated = self.trainer.generate_text(self.model, seed_token, max_len=5)
            self.assertIsInstance(generated, str)
            self.assertGreater(len(generated), 0)
            # Should start with seed token
            self.assertTrue(generated.startswith(seed_token))
        except Exception as e:
            if "data files not found" in str(e).lower() or "vocabulary" in str(e).lower():
                self.skipTest("Data files not found, skipping text generation test")
            else:
                raise

    def test_training_with_different_batch_sizes(self):
        """Test training with different batch sizes"""
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            vocab_size = self.vocab_service.get_vocab_size()
            config = ModelConfig(vocab_size=vocab_size, embed_dim=64, n_heads=2, n_layers=1)
            model = SmallTransformer(config)

            # Create dummy data using valid token IDs from vocabulary
            vocab = self.vocab_service.get_vocab()
            valid_token_ids = [tid for tid in vocab.values() if tid > 0]  # Skip PAD token (0)

            # Create test sentences with valid token IDs
            test_sentences = []
            for _ in range(3):  # Create 3 sentences
                # Each sentence needs at least 2 tokens for x,y split
                sentence_length = random.randint(3, 5)
                sentence = [random.choice(valid_token_ids) for _ in range(sentence_length)]
                test_sentences.append(sentence)

            dataset = SentenceDataset(test_sentences)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

            # Test one training step
            optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            try:
                x, y, mask = next(iter(loader))
                optimiser.zero_grad()
                logits = model(x, mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimiser.step()

                self.assertFalse(torch.isnan(loss))
            except StopIteration:
                # Empty loader
                pass

    def test_training_with_different_learning_rates(self):
        """Test training with different learning rates"""
        learning_rates = [0.001, 0.01, 0.1]

        for lr in learning_rates:
            config = ModelConfig(vocab_size=self.vocab_service.get_vocab_size(), embed_dim=64, n_heads=2, n_layers=1)
            model = SmallTransformer(config)

            # Create dummy data
            vocab_size = self.vocab_service.get_vocab_size()
            x = torch.randint(0, vocab_size, (1, 3))
            mask = torch.ones(1, 3, dtype=torch.bool)
            targets = torch.randint(0, vocab_size, (1, 3))

            # Store initial parameters
            initial_params = [p.clone() for p in model.parameters()]

            # Training step
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            optimiser.zero_grad()
            logits = model(x, mask)
            loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimiser.step()

            # Check that parameters changed
            final_params = list(model.parameters())
            params_changed = any(not torch.equal(init, final) for init, final in zip(initial_params, final_params))
            self.assertTrue(params_changed, f"Parameters didn't change with lr={lr}")

    def test_vocab_and_tokenisation(self):
        """Test vocabulary loading and tokenisation"""
        try:
            # Test that vocab is loaded
            vocab = self.vocab_service.get_vocab()
            self.assertIsInstance(vocab, dict)

            # Test tokenisation
            test_token = list(vocab.keys())[0]
            token_id = vocab[test_token]
            reconstructed_token = self.vocab_service.get_token_by_id(token_id)
            self.assertEqual(test_token, reconstructed_token)

            # Test <PAD> token
            self.assertEqual(self.vocab_service.get_token_by_id(0), "<PAD>")

        except Exception as e:
            if "data/tokens.txt" in str(e):
                self.skipTest("Vocabulary file not found, skipping vocab test")
            else:
                raise


if __name__ == "__main__":
    unittest.main()
