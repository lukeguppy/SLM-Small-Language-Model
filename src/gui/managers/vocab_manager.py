from ...services.vocab_service import VocabService


class VocabManager:
    """
    GUI Manager for vocabulary operations.
    Wraps VocabService with GUI-specific functionality.
    """

    def __init__(self, logger):
        self.vocab_service = VocabService(logger=logger)

    def load_vocabulary(self, vocab_path):
        """Load vocabulary from file path"""
        return self.vocab_service.load_vocabulary(vocab_path)

    def get_vocab(self):
        """Get the vocabulary dictionary"""
        return self.vocab_service.get_vocab()

    def get_id_to_token(self):
        """Get the id to token mapping"""
        return self.vocab_service.get_id_to_token()

    def get_vocab_size(self):
        """Get the vocabulary size"""
        return self.vocab_service.get_vocab_size()

    def text_to_tokens(self, text):
        """Convert text to token IDs"""
        return self.vocab_service.text_to_tokens(text)

    def tokens_to_text(self, tokens):
        """Convert token IDs to text"""
        return self.vocab_service.tokens_to_text(tokens)

    def is_token_valid(self, token):
        """Check if a token is in the vocabulary"""
        return self.vocab_service.is_token_valid(token)

    def get_token_suggestions(self, prefix, max_suggestions=10):
        """Get token suggestions for a prefix"""
        return self.vocab_service.get_token_suggestions(prefix, max_suggestions)

    # GUI-specific methods that delegate to service
    def get_token_id(self, token):
        """Get token ID (GUI convenience method)"""
        return self.vocab_service.get_token_id(token)

    def get_token_by_id(self, token_id):
        """Get token by ID (GUI convenience method)"""
        return self.vocab_service.get_token_by_id(token_id)

    # Service access for operations
    def get_vocab_service(self):
        """Get the underlying VocabService for operations"""
        return self.vocab_service
