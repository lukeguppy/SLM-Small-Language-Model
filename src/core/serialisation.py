import torch
import os
from typing import Any, Dict, Optional, Union


class SerialisationService:
    """
    Centralised service for PyTorch serialisation operations.

    Provides consistent error handling and device management for
    loading and saving PyTorch models and tensors.
    """

    @staticmethod
    def load_model_state(model_path: str, device: Union[str, torch.device], logger) -> Dict[str, Any]:
        """Load PyTorch model state dictionary with error handling."""

        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.debug(f"Loading model state from {model_path} to device {device}")
            state_dict = torch.load(model_path, map_location=device)

            if not isinstance(state_dict, dict):
                error_msg = f"Expected state_dict to be a dictionary, got {type(state_dict)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.debug(f"Successfully loaded state dict with {len(state_dict)} keys")
            return state_dict

        except Exception as e:
            error_msg = f"Failed to load model state from {model_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def save_model_state(state_dict: Dict[str, Any], model_path: str, logger) -> None:
        """Save PyTorch model state dictionary with error handling."""

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            logger.debug(f"Saving model state to {model_path}")
            torch.save(state_dict, model_path)
            logger.debug(f"Successfully saved model state to {model_path}")

        except Exception as e:
            error_msg = f"Failed to save model state to {model_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def load_tensor(tensor_path: str, device: Union[str, torch.device], logger) -> torch.Tensor:
        """Load PyTorch tensor with error handling."""

        if not os.path.exists(tensor_path):
            error_msg = f"Tensor file not found: {tensor_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.debug(f"Loading tensor from {tensor_path} to device {device}")
            tensor = torch.load(tensor_path, map_location=device)

            if not isinstance(tensor, torch.Tensor):
                error_msg = f"Expected tensor, got {type(tensor)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.debug(f"Successfully loaded tensor with shape {tensor.shape}")
            return tensor

        except Exception as e:
            error_msg = f"Failed to load tensor from {tensor_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def save_tensor(tensor: torch.Tensor, tensor_path: str, logger) -> None:
        """Save PyTorch tensor with error handling."""

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(tensor_path), exist_ok=True)

            logger.debug(f"Saving tensor to {tensor_path}")
            torch.save(tensor, tensor_path)
            logger.debug(f"Successfully saved tensor to {tensor_path}")

        except Exception as e:
            error_msg = f"Failed to save tensor to {tensor_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def validate_state_dict(state_dict: Dict[str, Any], logger) -> bool:
        """Validate that a state dictionary contains expected PyTorch parameters."""

        if not isinstance(state_dict, dict):
            logger.error("State dict is not a dictionary")
            return False

        if len(state_dict) == 0:
            logger.warning("State dict is empty")
            return False

        # Check that all values are tensors
        non_tensor_keys = []
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                non_tensor_keys.append(key)

        if non_tensor_keys:
            logger.warning(f"Found non-tensor values for keys: {non_tensor_keys[:5]}...")
            return False

        logger.debug(f"State dict validation passed: {len(state_dict)} tensor parameters")
        return True
