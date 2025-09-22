import torch
import numpy as np
from typing import Optional, Union, Tuple, List

from ..core.model import SmallTransformer
from ..model_config import ModelConfig
from ..core.model_loader import ModelLoader


class ModelService:
    """
    Service for model lifecycle management.
    Handles model loading, prediction, and analysis.
    """

    def __init__(self, logger=None):
        if logger is None:
            raise ValueError("logger parameter is required")
        self.logger = logger
        self.model: Optional[SmallTransformer] = None
        self.config: Optional[ModelConfig] = None

    def load_model(self, model_path: str, config: Optional[ModelConfig] = None, meta_path: Optional[str] = None) -> bool:
        """Load model from file path."""
        try:
            # Load model and config
            self.model, self.config = ModelLoader.load_model_safely(
                model_path=model_path, config=config, meta_path=meta_path, logger=self.logger
            )

            if self.model is None:
                return False

            self.model.eval()
            return True

        except Exception as e:
            print(f"Model load error: {str(e)}")
            return False

    def get_model(self) -> Optional[SmallTransformer]:
        """Get the loaded model."""
        return self.model

    def get_config(self) -> Optional[ModelConfig]:
        """Get the model configuration."""
        return self.config

    def predict_next_tokens(self, tokens: List[int], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next tokens for given token sequence."""
        if self.model is None or self.config is None:
            return []

        try:
            token_ids_tensor = torch.tensor([tokens], dtype=torch.long)

            with torch.no_grad():
                logits = self.model(token_ids_tensor)
                next_logits = logits[0, -1].cpu().numpy()

            # Get top k predictions
            top_indices = np.argsort(next_logits)[-top_k:][::-1]
            predictions = []

            for idx in top_indices:
                confidence = float(next_logits[idx])
                predictions.append((str(idx), confidence))

            return predictions

        except Exception as e:
            print(f"Prediction error: {e}")
            return []

    def get_attention_weights(self, tokens: List[int]) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        """Get attention weights for given token sequence."""
        if self.model is None:
            return None

        try:
            token_ids_tensor = torch.tensor([tokens], dtype=torch.long)

            with torch.no_grad():
                if hasattr(self.model, "forward"):
                    _, attention_weights = self.model.forward(
                        token_ids_tensor, return_attention=True, return_embeddings=False
                    )
                    return attention_weights

        except Exception as e:
            print(f"Attention extraction error: {e}")

        return None

    def get_embeddings(self, tokens: List[int]) -> Optional[torch.Tensor]:
        """Get embeddings for given token sequence."""
        if self.model is None:
            return None

        try:
            token_ids_tensor = torch.tensor([tokens], dtype=torch.long)

            with torch.no_grad():
                if hasattr(self.model, "forward"):
                    embeddings = self.model.forward(token_ids_tensor, return_attention=False, return_embeddings=True)
                    return embeddings.squeeze(0)  # Remove batch dimension

        except Exception as e:
            print(f"Embedding extraction error: {e}")

        return None

    def generate_text(self, seed_tokens: List[int], max_length: int = 10) -> List[int]:
        """Generate text continuation from seed tokens."""
        if self.model is None:
            return seed_tokens

        try:
            generated = seed_tokens.copy()
            model_device = next(self.model.parameters()).device

            for _ in range(max_length):
                # Prepare input
                input_tensor = torch.tensor([generated], dtype=torch.long).to(model_device)
                mask = torch.ones(1, len(generated), dtype=torch.bool).to(model_device)

                with torch.no_grad():
                    logits = self.model(input_tensor, mask)
                    next_token_logits = logits[0, -1]

                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                generated.append(next_token)

                # Stop if padding token is generated (assuming 0 is PAD)
                if next_token == 0:
                    break

            return generated

        except Exception as e:
            print(f"Text generation error: {e}")
            return seed_tokens

    @staticmethod
    def generate_text_from_seed(seed_word: str, max_length: int = 10, vocab_service=None, model_service=None) -> str:
        """Generate text continuation from seed word."""
        from .vocab_service import VocabService

        # Create services if not provided
        if vocab_service is None:
            vocab_service = VocabService()
        if model_service is None:
            model_service = ModelService()

        # Load vocabulary if not already loaded
        if vocab_service.get_vocab_size() == 0:
            from ..core.paths import DataPaths

            vocab_path = DataPaths.get_vocab_path()
            if not vocab_service.load_vocabulary(vocab_path):
                return f"Error: Could not load vocabulary from {vocab_path}"

        # Generate text using the model service
        seed_tokens = vocab_service.text_to_tokens(seed_word)
        if not seed_tokens:
            return f"Error: Could not tokenise seed word '{seed_word}'"

        generated_tokens = model_service.generate_text(seed_tokens, max_length)
        return vocab_service.tokens_to_text(generated_tokens)

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.config = None

    def get_model_info(self) -> Optional[dict]:
        """Get information about the loaded model."""
        if self.model is None or self.config is None:
            return None

        return {
            "vocab_size": self.config.vocab_size,
            "embed_dim": self.config.embed_dim,
            "n_heads": self.config.n_heads,
            "n_layers": self.config.n_layers,
            "model_path": getattr(self.config, "model_path", None),
            "saved_date": getattr(self.config, "saved_date", None),
            "saved_time": getattr(self.config, "saved_time", None),
        }

    def save_model_checkpoint(self, model_name: str, checkpoint_type: str, config=None) -> bool:
        """Save a model checkpoint with proper file management."""
        if self.model is None:
            return False

        try:
            from ..core.model_loader import ModelLoader
            from ..core.paths import ModelPaths

            # Determine paths based on checkpoint type
            if checkpoint_type == "best":
                model_path = ModelPaths.get_best_model_path(model_name)
                meta_path = ModelPaths.get_best_meta_path(model_name)
            elif checkpoint_type == "final":
                model_path = ModelPaths.get_final_model_path(model_name)
                meta_path = ModelPaths.get_final_meta_path(model_name)
            else:  # main
                model_path = ModelPaths.get_model_path(model_name)
                meta_path = ModelPaths.get_meta_path(model_name)

            # Use config from service if not provided
            save_config = config or self.config

            # Ensure save_config is a ModelConfig instance
            if not isinstance(save_config, ModelConfig):
                save_config = ModelConfig()

            # Save using ModelLoader
            ModelLoader.save_model_with_meta(self.model, model_path, self.logger, save_config, meta_path)

            return True

        except Exception as e:
            print(f"Failed to save {checkpoint_type} model: {e}")
            return False

    def cleanup_model_files(self, model_name: str, final_is_best: bool) -> None:
        """Clean up model files based on training completion logic."""
        from ..core.paths import ModelPaths
        import os

        if final_is_best:
            # Final model is best - save as main model and remove best/final files
            best_path = ModelPaths.get_best_model_path(model_name)
            final_path = ModelPaths.get_final_model_path(model_name)
            main_path = ModelPaths.get_model_path(model_name)

            # Copy best model to main (if it exists)
            if os.path.exists(best_path):
                import shutil

                shutil.copy2(best_path, main_path)

                # Remove best and final files
                for path in [best_path, final_path]:
                    if os.path.exists(path):
                        os.remove(path)

                # Also remove meta files
                best_meta = ModelPaths.get_best_meta_path(model_name)
                final_meta = ModelPaths.get_final_meta_path(model_name)
                for meta_path in [best_meta, final_meta]:
                    if os.path.exists(meta_path):
                        os.remove(meta_path)

        else:
            # Final model is not best - keep best and final, don't create main
            pass
