import re
from typing import List, Dict, Optional


class TokenValidator:
    """
    Token validation and processing utilities.

    Provides consistent token validation, filtering, and processing
    logic used by VocabService and DataService.
    """

    # Default validation parameters
    DEFAULT_MAX_TOKEN_LENGTH = 50
    DEFAULT_MIN_TOKEN_LENGTH = 1
    DEFAULT_MAX_SENTENCE_LENGTH = 100
    DEFAULT_MIN_SENTENCE_LENGTH = 2

    def __init__(
        self,
        max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
        min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
        max_sentence_length: int = DEFAULT_MAX_SENTENCE_LENGTH,
        min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
        allow_punctuation: bool = True,
        allow_numbers: bool = True,
        logger=None,
    ):
        """Initialise the token validator with custom parameters."""
        self.max_token_length = max_token_length
        self.min_token_length = min_token_length
        self.max_sentence_length = max_sentence_length
        self.min_sentence_length = min_sentence_length
        self.allow_punctuation = allow_punctuation
        self.allow_numbers = allow_numbers

        if logger is None:
            raise ValueError("logger parameter is required")
        self.logger = logger

        # Compile regex patterns for efficiency
        if not self.allow_punctuation:
            self._punct_pattern = re.compile(r"[^\w\s]")
        else:
            self._punct_pattern = None

        if not self.allow_numbers:
            self._number_pattern = re.compile(r"\d")
        else:
            self._number_pattern = None

    def validate_token(self, token: str) -> bool:
        """Validate a single token according to the configured rules."""
        if not isinstance(token, str):
            return False

        token = token.strip()

        # Check length
        if len(token) < self.min_token_length or len(token) > self.max_token_length:
            return False

        # Check for empty tokens
        if not token:
            return False

        # Check punctuation
        if self._punct_pattern and self._punct_pattern.search(token):
            return False

        # Check numbers
        if self._number_pattern and self._number_pattern.search(token):
            return False

        return True

    def validate_sentence_tokens(self, tokens: List[str]) -> bool:
        """Validate a list of tokens representing a sentence."""
        if not isinstance(tokens, list):
            return False

        # Check sentence length
        if len(tokens) < self.min_sentence_length or len(tokens) > self.max_sentence_length:
            return False

        # Check that all tokens are valid
        return all(self.validate_token(token) for token in tokens)

    def filter_valid_tokens(self, tokens: List[str]) -> List[str]:
        """Filter a list of tokens, keeping only valid ones."""
        if not isinstance(tokens, list):
            return []

        valid_tokens = [token for token in tokens if self.validate_token(token)]

        if len(valid_tokens) != len(tokens):
            invalid_count = len(tokens) - len(valid_tokens)
            self.logger.debug(f"Filtered out {invalid_count} invalid tokens")

        return valid_tokens

    def text_to_valid_tokens(self, text: str, vocab: Optional[Dict[str, int]] = None) -> List[str]:
        """Convert text to tokens and filter out invalid ones."""
        if not isinstance(text, str):
            return []

        # Basic tokenisation with punctuation stripping
        import re
        # Remove punctuation and split
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        raw_tokens = clean_text.split()

        # Filter valid tokens
        valid_tokens = self.filter_valid_tokens(raw_tokens)

        # Additional vocab filtering if provided
        if vocab is not None:
            vocab_tokens = [token for token in valid_tokens if token in vocab]
            if len(vocab_tokens) != len(valid_tokens):
                filtered_count = len(valid_tokens) - len(vocab_tokens)
                self.logger.debug(f"Filtered out {filtered_count} tokens not in vocabulary")
            valid_tokens = vocab_tokens

        return valid_tokens

    def tokens_to_ids(self, tokens: List[str], vocab: Dict[str, int], unknown_token_id: int = 0) -> List[int]:
        """Convert tokens to their corresponding IDs using vocabulary."""
        if not isinstance(tokens, list) or not isinstance(vocab, dict):
            return []

        token_ids = []
        for token in tokens:
            token_id = vocab.get(token, unknown_token_id)
            token_ids.append(token_id)

        return token_ids

    def filter_sentences_by_vocab(self, sentences: List[str], vocab: Dict[str, int]) -> List[List[int]]:
        """Filter sentences to only include words in vocabulary and convert to token IDs."""
        if not isinstance(sentences, list) or not isinstance(vocab, dict):
            return []

        tokenised_sentences = []

        for sentence in sentences:
            # Convert sentence to valid tokens
            tokens = self.text_to_valid_tokens(sentence, vocab)

            # Convert tokens to IDs
            if tokens:  # Only add non-empty sentences
                token_ids = self.tokens_to_ids(tokens, vocab)
                if len(token_ids) >= self.min_sentence_length:
                    tokenised_sentences.append(token_ids)

        filtered_count = len(sentences) - len(tokenised_sentences)
        if filtered_count > 0:
            self.logger.debug(f"Filtered out {filtered_count} sentences that were too short or empty")

        return tokenised_sentences

    def get_validation_stats(self, tokens: List[str]) -> Dict[str, int]:
        """Get statistics about token validation."""
        if not isinstance(tokens, list):
            return {"total": 0, "valid": 0, "invalid": 0}

        total = len(tokens)
        valid = len(self.filter_valid_tokens(tokens))
        invalid = total - valid

        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "valid_percentage": (valid / total * 100) if total > 0 else 0,
        }

    def validate_vocabulary_coverage(self, tokens: List[str], vocab: Dict[str, int]) -> Dict[str, float]:
        """Validate vocabulary coverage for a list of tokens."""
        if not isinstance(tokens, list) or not isinstance(vocab, dict):
            return {"coverage": 0.0, "unknown": 1.0}

        total = len(tokens)
        if total == 0:
            return {"coverage": 0.0, "unknown": 0.0}

        known = sum(1 for token in tokens if token in vocab)
        unknown = total - known

        return {"coverage": known / total, "unknown": unknown / total, "known_tokens": known, "unknown_tokens": unknown}
