import os
import random
from typing import List, Tuple, Optional

from ..core.paths import DataPaths


class DataService:
    """
    Service for data loading and preprocessing.
    Handles vocabulary and sentence data loading.
    """

    def __init__(self, logger=None):
        if logger is None:
            raise ValueError("logger parameter is required")
        self.logger = logger
        self.sentences: List[str] = []
        self.vocab_path: Optional[str] = None
        self.sentences_path: Optional[str] = None

    def load_data(self, vocab_path: Optional[str] = None, sentences_path: Optional[str] = None) -> bool:
        """Load vocabulary and sentence data."""
        try:
            # Use provided paths or defaults
            self.vocab_path = vocab_path or DataPaths.get_vocab_path()
            self.sentences_path = sentences_path or DataPaths.get_sentences_path()

            # Load sentences
            if os.path.exists(self.sentences_path):
                with open(self.sentences_path, "r", encoding="utf-8") as f:
                    self.sentences = [line.strip() for line in f if line.strip()]
            else:
                raise FileNotFoundError(f"Sentences file not found: {self.sentences_path}")

            return True

        except Exception as e:
            print(f"Data loading error: {e}")
            return False

    def get_sentences(self) -> List[str]:
        """Get loaded sentences."""
        return self.sentences.copy()

    def get_sentence_count(self) -> int:
        """Get number of loaded sentences."""
        return len(self.sentences)

    def get_vocab_path(self) -> Optional[str]:
        """Get the vocabulary file path."""
        return self.vocab_path

    def get_sentences_path(self) -> Optional[str]:
        """Get the sentences file path."""
        return self.sentences_path

    def filter_sentences_by_vocab(self, vocab: dict) -> List[List[int]]:
        """Filter sentences to only include words in vocabulary using TokenValidator."""
        from ..core.token_validator import TokenValidator

        # Use TokenValidator for consistent processing
        validator = TokenValidator(logger=self.logger)
        return validator.filter_sentences_by_vocab(self.sentences, vocab)

    def create_train_val_test_split(
        self,
        tokenised_sentences: List[List[int]],
        train_size: int = 50000,
        val_size: int = 10000,
        test_size: int = 10000,
        augment: bool = True,
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Create train/validation/test split from tokenised sentences."""
        # Shuffle the data
        shuffled_sentences = tokenised_sentences.copy()
        random.shuffle(shuffled_sentences)

        # Split the data
        base_train = shuffled_sentences[:train_size]
        val_sentences = shuffled_sentences[train_size : train_size + val_size]
        test_sentences = shuffled_sentences[train_size + val_size : train_size + val_size + test_size]

        # Augment training data if requested
        if augment:
            train_sentences = []
            for sentence in base_train:
                train_sentences.append(sentence)
                if random.random() < 0.3:  # 30% chance to augment (adjusts data to make it more diverse)
                    train_sentences.append(self._augment_sentence(sentence.copy()))
        else:
            train_sentences = base_train

        return train_sentences, val_sentences, test_sentences

    def _augment_sentence(self, tokens: List[int]) -> List[int]:
        """Apply data augmentation to a token sequence."""
        # No augmentation applied - sentences kept as-is for linguistic coherence
        # Any augmentation (sentence noise or alterations) added here
        return tokens

    def get_data_statistics(self) -> dict:
        """Get statistics about the loaded data."""
        if not self.sentences:
            return {
                "sentence_count": 0,
                "total_words": 0,
                "avg_words_per_sentence": 0,
                "max_words_per_sentence": 0,
                "min_words_per_sentence": 0,
            }

        word_counts = [len(sentence.split()) for sentence in self.sentences]

        return {
            "sentence_count": len(self.sentences),
            "total_words": sum(word_counts),
            "avg_words_per_sentence": sum(word_counts) / len(self.sentences),
            "max_words_per_sentence": max(word_counts),
            "min_words_per_sentence": min(word_counts),
        }

    def validate_data(self) -> List[str]:
        """Validate the loaded data and return any issues found."""
        issues = []

        if not self.sentences:
            issues.append("No sentences loaded")
            return issues

        # Check for empty sentences
        empty_count = sum(1 for s in self.sentences if not s.strip())
        if empty_count > 0:
            issues.append(f"Found {empty_count} empty sentences")

        # Check for very short sentences
        short_count = sum(1 for s in self.sentences if len(s.split()) < 2)
        if short_count > 0:
            issues.append(f"Found {short_count} sentences with fewer than 2 words")

        # Check for very long sentences
        long_count = sum(1 for s in self.sentences if len(s.split()) > 100)
        if long_count > 0:
            issues.append(f"Found {long_count} sentences with more than 100 words")

        return issues

    @staticmethod
    def create_data_loaders(
        batch_size=16,
        train_size=50000,
        val_size=10000,
        test_size=10000,
        vocab_path="data/words.txt",
        sentences_path=None,
    ):
        """Create data loaders for training, validation, and testing."""
        from .vocab_service import VocabService
        from ..training.trainer import SentenceDataset, collate_fn
        from torch.utils.data import DataLoader

        # Create services
        vocab_service = VocabService()
        data_service = DataService()

        # Load data
        if not vocab_service.load_vocabulary(vocab_path):
            raise RuntimeError("Failed to load vocabulary")

        if not data_service.load_data(vocab_path=vocab_path, sentences_path=sentences_path):
            raise RuntimeError("Failed to load data")

        vocab = vocab_service.get_vocab()
        tokenised_sentences = data_service.filter_sentences_by_vocab(vocab)

        # Create splits
        train_sentences, val_sentences, test_sentences = data_service.create_train_val_test_split(
            tokenised_sentences, train_size, val_size, test_size, augment=True
        )

        # Create loaders
        train_loader = DataLoader(
            SentenceDataset(train_sentences), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            SentenceDataset(val_sentences), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            SentenceDataset(test_sentences), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        return train_loader, val_loader, test_loader
