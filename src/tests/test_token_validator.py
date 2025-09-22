import unittest
from ..core.token_validator import TokenValidator
from ..core.logger import get_testing_logger


class TestTokenValidator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        logger = get_testing_logger()
        self.validator = TokenValidator(logger=logger)
        self.strict_validator = TokenValidator(
            max_token_length=5, min_token_length=2, allow_punctuation=False, allow_numbers=False, logger=logger
        )

    def test_validate_token_basic(self):
        """Test basic token validation."""
        # Valid tokens
        self.assertTrue(self.validator.validate_token("hello"))
        self.assertTrue(self.validator.validate_token("world"))
        self.assertTrue(self.validator.validate_token("test123"))

        # Invalid tokens
        self.assertFalse(self.validator.validate_token(""))  # Empty
        self.assertFalse(self.validator.validate_token("a" * 100))  # Too long

    def test_validate_token_length_constraints(self):
        """Test token validation with length constraints."""
        # Test minimum length
        self.assertFalse(self.strict_validator.validate_token("a"))  # Too short
        self.assertTrue(self.strict_validator.validate_token("ab"))  # Minimum
        self.assertTrue(self.strict_validator.validate_token("hello"))  # Valid

        # Test maximum length
        self.assertTrue(self.strict_validator.validate_token("hello"))  # Valid
        self.assertFalse(self.strict_validator.validate_token("helloworld"))  # Too long

    def test_validate_token_punctuation(self):
        """Test token validation with punctuation constraints."""
        # With punctuation allowed
        self.assertTrue(self.validator.validate_token("hello!"))
        self.assertTrue(self.validator.validate_token("world?"))

        # With punctuation not allowed
        self.assertFalse(self.strict_validator.validate_token("hello!"))
        self.assertFalse(self.strict_validator.validate_token("world?"))

    def test_validate_token_numbers(self):
        """Test token validation with number constraints."""
        # With numbers allowed
        self.assertTrue(self.validator.validate_token("test123"))
        self.assertTrue(self.validator.validate_token("42"))

        # With numbers not allowed
        self.assertFalse(self.strict_validator.validate_token("test123"))
        self.assertFalse(self.strict_validator.validate_token("42"))

    def test_validate_sentence_tokens(self):
        """Test sentence token validation."""
        # Valid sentence
        valid_tokens = ["hello", "world", "test"]
        self.assertTrue(self.validator.validate_sentence_tokens(valid_tokens))

        # Invalid sentence - empty
        self.assertFalse(self.validator.validate_sentence_tokens([]))

        # Invalid sentence - contains invalid token
        invalid_tokens = ["hello", "", "world"]  # Empty token
        self.assertFalse(self.validator.validate_sentence_tokens(invalid_tokens))

    def test_filter_valid_tokens(self):
        """Test filtering valid tokens from a list."""
        tokens = ["hello", "", "world", "a" * 100, "test"]
        filtered = self.validator.filter_valid_tokens(tokens)

        # Should only keep "hello", "world", "test"
        self.assertEqual(filtered, ["hello", "world", "test"])

    def test_text_to_valid_tokens(self):
        """Test converting text to valid tokens."""
        text = "Hello, world! This is a test."
        vocab = {"hello": 0, "world": 1, "this": 2, "is": 3, "a": 4, "test": 5}

        tokens = self.validator.text_to_valid_tokens(text, vocab)
        expected = ["hello", "world", "this", "is", "a", "test"]
        self.assertEqual(tokens, expected)

    def test_text_to_valid_tokens_without_vocab(self):
        """Test converting text to valid tokens without vocab filtering."""
        text = "Hello, world! This is a test."
        tokens = self.validator.text_to_valid_tokens(text)

        # Should include all valid tokens (punctuation is stripped)
        expected = ["hello", "world", "this", "is", "a", "test"]
        self.assertEqual(tokens, expected)

    def test_tokens_to_ids(self):
        """Test converting tokens to IDs."""
        tokens = ["hello", "world", "unknown"]
        vocab = {"hello": 1, "world": 2}

        token_ids = self.validator.tokens_to_ids(tokens, vocab)
        expected = [1, 2, 0]  # unknown defaults to 0
        self.assertEqual(token_ids, expected)

    def test_filter_sentences_by_vocab(self):
        """Test filtering sentences by vocabulary."""
        sentences = ["hello world", "unknown word", "test sentence"]
        vocab = {"hello": 0, "world": 1, "test": 2, "sentence": 3}

        filtered = self.validator.filter_sentences_by_vocab(sentences, vocab)

        # Should only keep sentences with valid tokens
        self.assertEqual(len(filtered), 2)  # "hello world" and "test sentence"
        self.assertEqual(filtered[0], [0, 1])  # hello world -> IDs
        self.assertEqual(filtered[1], [2, 3])  # test sentence -> IDs

    def test_get_validation_stats(self):
        """Test getting validation statistics."""
        tokens = ["hello", "", "world", "a" * 100]
        stats = self.validator.get_validation_stats(tokens)

        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["valid"], 2)  # hello, world
        self.assertEqual(stats["invalid"], 2)  # empty, too long

    def test_validate_vocabulary_coverage(self):
        """Test vocabulary coverage validation."""
        tokens = ["hello", "world", "unknown"]
        vocab = {"hello": 0, "world": 1}

        coverage = self.validator.validate_vocabulary_coverage(tokens, vocab)

        self.assertAlmostEqual(coverage["coverage"], 2 / 3, places=2)
        self.assertAlmostEqual(coverage["unknown"], 1 / 3, places=2)
        self.assertEqual(coverage["known_tokens"], 2)
        self.assertEqual(coverage["unknown_tokens"], 1)

    def test_custom_validator_parameters(self):
        """Test validator with custom parameters."""
        logger = get_testing_logger()
        custom_validator = TokenValidator(
            max_token_length=10,
            min_token_length=3,
            max_sentence_length=5,
            min_sentence_length=2,
            allow_punctuation=False,
            allow_numbers=False,
            logger=logger,
        )

        # Test constraints
        self.assertTrue(custom_validator.validate_token("hello"))  # Valid
        self.assertFalse(custom_validator.validate_token("hi"))  # Too short
        self.assertFalse(custom_validator.validate_token("hello!"))  # Punctuation not allowed
        self.assertFalse(custom_validator.validate_token("hello123"))  # Numbers not allowed


if __name__ == "__main__":
    unittest.main()
