import os
from typing import Dict, List, Optional


class VocabService:
    """
    Service for vocabulary loading and management.
    Handles token-to-ID mapping and text tokenisation.
    """

    def __init__(self, logger=None):
        if logger is None:
            raise ValueError("logger parameter is required")
        self.logger = logger
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

    def load_vocabulary(self, vocab_path: str) -> bool:
        """Load vocabulary from file path."""
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                tokens = [line.strip() for line in f if line.strip()]

            # Create vocabulary mappings
            self.vocab = {token: i + 1 for i, token in enumerate(tokens)}  # Shift to 1+
            self.vocab["<PAD>"] = 0
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.vocab_size = len(tokens) + 1  # +1 for <PAD>

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to load vocabulary from {vocab_path}: {e}")

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary."""
        return self.vocab.copy()

    def get_id_to_token(self) -> Dict[int, str]:
        """Get the ID to token mapping."""
        return self.id_to_token.copy()

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.vocab_size

    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs using TokenValidator for consistency."""
        from ..core.token_validator import TokenValidator

        # Use TokenValidator for consistent tokenisation
        validator = TokenValidator(logger=self.logger)
        tokens = validator.text_to_valid_tokens(text, self.vocab)

        # Convert to IDs
        token_ids = validator.tokens_to_ids(tokens, self.vocab, unknown_token_id=0)
        return token_ids

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text."""
        token_texts = [self.id_to_token.get(token_id, "<UNK>") for token_id in tokens]
        return " ".join(token_texts)

    def is_token_valid(self, token: str) -> bool:
        """Check if a token is in the vocabulary."""
        return token.lower() in self.vocab

    def get_token_suggestions(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get token suggestions for a prefix."""
        prefix = prefix.lower()
        matches = [token for token in self.vocab.keys() if token.lower().startswith(prefix)]
        return matches[:max_suggestions]

    def set_vocab(self, vocab: Dict[str, int]) -> None:
        """Set the vocabulary dictionary."""
        self.vocab = vocab.copy()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def set_id_to_token(self, id_to_token: Dict[int, str]) -> None:
        """Set the ID to token mapping."""
        self.id_to_token = id_to_token.copy()
        self.vocab = {v: k for k, v in self.id_to_token.items()}
        self.vocab_size = len(self.vocab)

    def set_vocab_size(self, vocab_size: int) -> None:
        """Set the vocabulary size."""
        self.vocab_size = vocab_size

    def get_token_id(self, token: str) -> Optional[int]:
        """Get the ID for a token."""
        return self.vocab.get(token.lower())

    def get_token_by_id(self, token_id: int) -> Optional[str]:
        """Get the token for an ID."""
        return self.id_to_token.get(token_id)
