import unittest
import re
import csv


class TestTokenisation(unittest.TestCase):
    def test_tokenise_sample_sentence(self):
        """Test tokenisation on a sample sentence"""
        # Sample sentence
        sentence = "Hello, world! This is a test sentence."

        # Tokenisation logic (from data processing pipeline)
        processed_sentence = sentence.lower()
        processed_sentence = re.sub(r"[^\w\s]", "", processed_sentence)
        tokens = processed_sentence.split()

        # Expected result
        expected_tokens = ["hello", "world", "this", "is", "a", "test", "sentence"]

        # Show the result
        print(f"Original sentence: '{sentence}'")
        print(f"Tokens: {tokens}")

        # Assert the result
        self.assertEqual(tokens, expected_tokens)

    def test_tokenise_from_sentences_file(self):
        """Test tokenisation on a sentence from the sentences file using training method"""
        try:
            # Try to read from the actual sentences file used by the trainer
            with open("data/sentences.txt", "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]

            if sentences:
                sentence = sentences[0]  # Use first sentence
            else:
                # Fallback to a test sentence if file is empty
                sentence = "hello world this is a test"

            # Tokenisation method used for training
            tokens = sentence.lower().split()

            # Show the result
            print(f"Original sentence from file: '{sentence}'")
            print(f"Tokens (training method): {tokens}")

            # Assert it's not empty
            self.assertGreater(len(tokens), 0)

        except FileNotFoundError:
            # Skip test if data files don't exist
            self.skipTest("Data files not found - skipping file-based tokenisation test")


if __name__ == "__main__":
    unittest.main()
