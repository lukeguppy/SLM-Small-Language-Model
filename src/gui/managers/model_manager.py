import os
import torch
import numpy as np
from typing import Optional, List, Union
from ...core.paths import ModelPaths
from ...services.model_service import ModelService


class ModelManager:
    """
    GUI Manager for model operations.
    Wraps ModelService with GUI-specific functionality.
    """

    def __init__(self, vocab_manager, config, logger):
        self.vocab_manager = vocab_manager
        self.config = config
        self.model_service = ModelService(logger=logger)

    def load_model(self, model_name):
        """Load model from file name using ModelService"""
        print(f"Loading model: {model_name}")
        if not model_name:
            return False

        # Handle paths that already contain directory structure
        if os.path.sep in model_name or (os.altsep and os.altsep in model_name):
            # This is already a path, extract the model name
            base_name = os.path.basename(model_name)
            if base_name.endswith(".pt"):
                model_name_clean = base_name[:-3]
            else:
                model_name_clean = base_name
            model_path = ModelPaths.get_model_path(model_name_clean)
            meta_path = ModelPaths.get_meta_path(model_name_clean)
        else:
            # Handle best and final variants
            if model_name.endswith("_best"):
                base = model_name[:-5]
                model_path = ModelPaths.get_best_model_path(base)
                meta_path = ModelPaths.get_best_meta_path(base)
            elif model_name.endswith("_final"):
                base = model_name[:-6]
                model_path = ModelPaths.get_final_model_path(base)
                meta_path = ModelPaths.get_final_meta_path(base)
            else:
                model_path = ModelPaths.get_model_path(model_name)
                meta_path = ModelPaths.get_meta_path(model_name)

        try:
            # Clean up previous model to prevent memory issues
            if self.model_service.get_model() is not None:
                self.model_service.unload_model()

            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False

            # Use ModelService for loading
            success = self.model_service.load_model(
                model_path, self.config, meta_path=meta_path
            )

            if not success:
                print("ModelService failed to load the model")
                return False

            print("Model loaded successfully")
            return True

        except Exception as e:
            print(f"Model load error: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def predict_next_tokens(self, text, top_k=5):
        """Predict next tokens for given text"""
        if not self.model_service.is_model_loaded():
            return []

        # Use vocab manager for tokenisation
        token_ids = self.vocab_manager.text_to_tokens(text)
        if not token_ids:
            return []

        # Use model service for prediction
        predictions = self.model_service.predict_next_tokens(token_ids, top_k)

        # Convert token IDs back to tokens using vocab manager
        result = []
        for token_id_str, confidence in predictions:
            try:
                token_id = int(token_id_str)
                token = self.vocab_manager.get_id_to_token().get(token_id, "<UNK>")
                result.append((token, confidence))
            except (ValueError, KeyError):
                result.append(("<UNK>", confidence))

        return result

    def get_attention_weights(
        self, text: str
    ) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        """Get attention weights for given text"""
        if not self.model_service.is_model_loaded():
            return None

        # Use vocab manager for tokenisation
        token_ids = self.vocab_manager.text_to_tokens(text)
        if not token_ids:
            return None

        # Use model service for attention extraction
        return self.model_service.get_attention_weights(token_ids)

    def get_embeddings(self, text):
        """Get embeddings for given text"""
        if not self.model_service.is_model_loaded():
            return None

        # Use vocab manager for tokenisation
        token_ids = self.vocab_manager.text_to_tokens(text)
        if not token_ids:
            return None

        # Use model service for embedding extraction
        return self.model_service.get_embeddings(token_ids)

    # Service access for advanced operations
    def get_model_service(self):
        """Get the underlying ModelService for advanced operations"""
        return self.model_service

    def get_model_info(self):
        """Get information about the loaded model"""
        return self.model_service.get_model_info()
