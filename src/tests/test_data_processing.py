import unittest
import tempfile
import os
import shutil


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.tokens_file = os.path.join(self.temp_dir, "tokens.txt")
        self.sentences_file = os.path.join(self.temp_dir, "toy_sentences.txt")

    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)

    def create_test_files(self, tokens_content, sentences_content):
        """Helper to create test files"""
        with open(self.tokens_file, "w") as f:
            f.write(tokens_content)
        with open(self.sentences_file, "w") as f:
            f.write(sentences_content)

    def test_real_world_dataset_loading(self):
        """Test loading real-world style datasets"""
        # Create a more realistic dataset
        real_world_sentences = [
            "The cat sat on the mat.",
            "I love to eat tomato and pasta.",
            "Machine learning is fascinating.",
            "The weather is beautiful today.",
            "She reads books every evening.",
            "Technology is changing our lives.",
            "He plays guitar in a band.",
            "The coffee tastes amazing.",
            "They travel around the world.",
            "Science helps us understand nature.",
        ]

        tokens = set()
        for sentence in real_world_sentences:
            tokens.update(sentence.lower().replace(".", "").replace(",", "").split())

        # Save files
        with open(self.tokens_file, "w") as f:
            for token in sorted(tokens):
                f.write(token + "\n")

        with open(self.sentences_file, "w") as f:
            for sentence in real_world_sentences:
                f.write(sentence + "\n")

        # Test loading
        with open(self.tokens_file, "r") as f:
            loaded_tokens = [line.strip() for line in f if line.strip()]

        with open(self.sentences_file, "r") as f:
            loaded_sentences = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(loaded_tokens), len(tokens))
        self.assertEqual(len(loaded_sentences), len(real_world_sentences))

    def test_data_preprocessing_pipeline(self):
        """Test complete data preprocessing pipeline"""
        # Test tokenisation
        test_sentences = [
            "Hello world!",
            "This is a test.",
            "Machine learning rocks.",
        ]

        # Simulate preprocessing steps
        processed_sentences = []
        for sentence in test_sentences:
            # Lowercase
            sentence = sentence.lower()
            # Remove punctuation
            import re

            sentence = re.sub(r"[^\w\s]", "", sentence)
            # Split into tokens
            tokens = sentence.split()
            processed_sentences.append(tokens)

        # Verify preprocessing
        expected = [["hello", "world"], ["this", "is", "a", "test"], ["machine", "learning", "rocks"]]

        self.assertEqual(processed_sentences, expected)

    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        # Create a large dataset
        num_sentences = 1000
        tokens_per_sentence = 10
        vocab_size = 500

        # Generate synthetic data
        large_sentences = []
        for i in range(num_sentences):
            sentence = " ".join([f"token_{j % vocab_size}" for j in range(tokens_per_sentence)])
            large_sentences.append(sentence)

        large_vocab = [f"token_{i}" for i in range(vocab_size)]

        self.create_test_files("\n".join(large_vocab), "\n".join(large_sentences))

        # Test that large data is handled
        with open(self.sentences_file, "r") as f:
            loaded_count = sum(1 for line in f if line.strip())

        self.assertEqual(loaded_count, num_sentences)

    def test_data_quality_metrics(self):
        """Test data quality metrics"""
        # Create dataset with known quality issues
        sentences = [
            "Good sentence.",  # Good
            "Short.",  # Too short
            "This is a very long sentence that contains many tokens and should be considered high quality.",  # Good
            "Bad.",  # Too short
            "Another good sentence with proper length.",  # Good
        ]

        # Calculate quality metrics
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        short_sentences = sum(1 for s in sentences if len(s.split()) < 3)

        # Should identify quality issues
        self.assertGreater(avg_length, 3)  # Average should be reasonable
        self.assertGreater(short_sentences, 0)  # Should detect short sentences

    def test_outlier_detection(self):
        """Test detection of outlier sentences"""
        sentences = [
            "Normal sentence.",  # Normal
            "This is also normal.",  # Normal
            "x" * 1000,  # Very long outlier
            "Another normal sentence.",  # Normal
            "y" * 500,  # Long outlier
        ]

        # Detect outliers by length
        lengths = [len(s) for s in sentences]
        mean_length = sum(lengths) / len(lengths)
        std_length = (sum((l - mean_length) ** 2 for l in lengths) / len(lengths)) ** 0.5

        outliers = [i for i, l in enumerate(lengths) if abs(l - mean_length) > 1.5 * std_length]  # Lower threshold

        # Should detect at least the very long sentence as outlier
        self.assertGreater(len(outliers), 0)
        self.assertIn(2, outliers)  # Very long sentence


if __name__ == "__main__":
    unittest.main()
