import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

from ..services.vocab_service import VocabService
from ..services.data_service import DataService
from ..core.token_validator import TokenValidator
from ..core.logger import get_testing_logger


class TestVocabService(unittest.TestCase):
    """Test VocabService functionality"""

    def setUp(self):
        self.logger = get_testing_logger()
        self.vocab_service = VocabService(logger=self.logger)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_vocabulary_success(self):
        """Test successful vocabulary loading"""
        vocab_content = "hello\nworld\ntest\n<PAD>\n"
        vocab_path = os.path.join(self.temp_dir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write(vocab_content)

        success = self.vocab_service.load_vocabulary(vocab_path)
        self.assertTrue(success)

        vocab = self.vocab_service.get_vocab()
        self.assertIsInstance(vocab, dict)
        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        self.assertIn("test", vocab)
        self.assertEqual(vocab["<PAD>"], 0)  # PAD token should be 0

    def test_load_vocabulary_file_not_found(self):
        """Test vocabulary loading when file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            self.vocab_service.load_vocabulary("/nonexistent/path/vocab.txt")

    def test_get_vocab_size(self):
        """Test getting vocabulary size"""
        vocab_content = "hello\nworld\ntest\n<PAD>\n"
        vocab_path = os.path.join(self.temp_dir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write(vocab_content)

        self.vocab_service.load_vocabulary(vocab_path)
        vocab_size = self.vocab_service.get_vocab_size()
        self.assertEqual(vocab_size, 5)  # 4 tokens + 1 for <PAD>

    def test_text_to_tokens(self):
        """Test text to tokens conversion"""
        vocab_content = "hello\nworld\ntest\n<PAD>\n"
        vocab_path = os.path.join(self.temp_dir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write(vocab_content)

        self.vocab_service.load_vocabulary(vocab_path)

        tokens = self.vocab_service.text_to_tokens("hello world")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_tokens_to_text(self):
        """Test tokens to text conversion"""
        vocab_content = "hello\nworld\ntest\n<PAD>\n"
        vocab_path = os.path.join(self.temp_dir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write(vocab_content)

        self.vocab_service.load_vocabulary(vocab_path)

        # Test with valid tokens
        tokens = [1, 2, 3]  # hello, world, test
        text = self.vocab_service.tokens_to_text(tokens)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


class TestDataService(unittest.TestCase):
    """Test DataService functionality"""

    def setUp(self):
        self.logger = get_testing_logger()
        self.data_service = DataService(logger=self.logger)
        self.vocab_service = VocabService(logger=self.logger)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_data_success(self):
        """Test successful data loading"""
        # Create test data files
        vocab_content = "hello\nworld\ntest\n<PAD>\n"
        vocab_path = os.path.join(self.temp_dir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write(vocab_content)

        sentences_content = "hello world\ntest sentence\nworld test\n"
        sentences_path = os.path.join(self.temp_dir, "sentences.txt")
        with open(sentences_path, "w") as f:
            f.write(sentences_content)

        # Mock the path manager to return our test files
        with patch("src.services.data_service.DataPaths") as mock_pm:
            mock_pm.get_vocab_path.return_value = vocab_path
            mock_pm.get_sentences_path.return_value = sentences_path

            success = self.data_service.load_data()
            self.assertTrue(success)

    def test_load_data_files_not_found(self):
        """Test data loading when files don't exist"""
        with patch("src.services.data_service.DataPaths") as mock_pm:
            mock_pm.get_vocab_path.return_value = "/nonexistent/vocab.txt"
            mock_pm.get_sentences_path.return_value = "/nonexistent/sentences.txt"

            success = self.data_service.load_data()
            self.assertFalse(success)

    def test_filter_sentences_by_vocab(self):
        """Test filtering sentences by vocabulary"""
        # Setup vocab
        vocab_content = "hello\nworld\ntest\n<PAD>\n"
        vocab_path = os.path.join(self.temp_dir, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write(vocab_content)

        self.vocab_service.load_vocabulary(vocab_path)
        vocab = self.vocab_service.get_vocab()

        # Setup sentences in data_service
        sentences_content = "hello world\ntest sentence\nworld test\n"
        sentences_path = os.path.join(self.temp_dir, "sentences.txt")
        with open(sentences_path, "w") as f:
            f.write(sentences_content)

        # Load data into data_service
        self.data_service.load_data(vocab_path, sentences_path)

        filtered = self.data_service.filter_sentences_by_vocab(vocab)
        self.assertIsInstance(filtered, list)
        # Should return tokenised sentences
        self.assertGreater(len(filtered), 0)


class TestTokenValidator(unittest.TestCase):
    """Test TokenValidator functionality"""

    def setUp(self):
        self.logger = get_testing_logger()
        self.validator = TokenValidator(logger=self.logger)

    def test_validate_tokens_valid(self):
        """Test validation of valid tokens"""
        vocab = {"hello": 1, "world": 2, "<PAD>": 0}
        tokens = ["hello", "world"]

        is_valid = all(self.validator.validate_token(token) for token in tokens)
        self.assertTrue(is_valid)

    def test_validate_tokens_invalid(self):
        """Test validation of invalid tokens"""
        tokens = ["hello", ""]  # Empty token is invalid

        is_valid = all(self.validator.validate_token(token) for token in tokens)
        self.assertFalse(is_valid)

    def test_validate_tokens_empty(self):
        """Test validation of empty tokens"""
        tokens = []

        is_valid = all(self.validator.validate_token(token) for token in tokens)
        self.assertTrue(is_valid)  # Empty list should be valid (no invalid tokens)

    def test_get_validation_stats(self):
        """Test getting validation statistics"""
        # Create a validator with strict length limits
        strict_validator = TokenValidator(max_token_length=10, logger=self.logger)
        tokens = ["hello", "world", "invalid_token_with_long_name"]

        stats = strict_validator.get_validation_stats(tokens)
        self.assertIsInstance(stats, dict)
        self.assertIn("total", stats)
        self.assertIn("valid", stats)
        self.assertIn("invalid", stats)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["valid"], 2)  # hello and world are valid
        self.assertEqual(stats["invalid"], 1)  # long token is invalid


if __name__ == "__main__":
    unittest.main()
