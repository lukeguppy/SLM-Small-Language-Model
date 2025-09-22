import torch
import os

from .model import SmallTransformer
from ..model_config import ModelConfig, ConfigPersistence
from .paths import ModelPaths


class ModelLoader:
    """Utility class to handle loading models with parameter mismatches."""

    @staticmethod
    def load_model_safely(model_path, config=None, vocab_size=None, meta_path=None, logger=None):
        """Load a model with robust error handling for parameter mismatches."""
        if logger is None:
            raise ValueError("logger parameter is required")

        # Setup configuration and paths
        config, meta_path, device = ModelLoader._setup_loading_config(model_path, config, vocab_size, meta_path, logger)

        # Load and prepare model
        try:
            model, config = ModelLoader._load_and_prepare_model(model_path, config, meta_path, device, logger)
            return model, config
        except Exception as e:
            ModelLoader._handle_loading_error(e, model_path, config, device, logger)
            raise

    @staticmethod
    def _setup_loading_config(model_path, config, vocab_size, meta_path, logger):
        """Setup configuration and determine paths for loading."""
        # Create default config if none provided
        if config is None:
            config = ModelConfig(vocab_size=vocab_size or 1000)

        # Set vocab_size if provided separately
        if vocab_size is not None:
            config.vocab_size = vocab_size

        # Determine meta path
        if meta_path is None:
            model_name = os.path.basename(model_path)
            if model_name.endswith(".pt"):
                model_name = model_name[:-3]
            # Use appropriate meta path based on model name
            if model_name.endswith("_best"):
                base_name = model_name[:-5]  # Remove '_best'
                meta_path = ModelPaths.get_best_meta_path(base_name)
            elif model_name.endswith("_final"):
                base_name = model_name[:-6]  # Remove '_final'
                meta_path = ModelPaths.get_final_meta_path(base_name)
            else:
                meta_path = ModelPaths.get_meta_path(model_name)

        # Try to load config from meta file
        logger.info(f"Looking for .meta file at: {meta_path}")
        loaded_config = ConfigPersistence.load_from_file(meta_path, logger)
        if loaded_config:
            config = loaded_config
            logger.info(f"Using parameters from .meta file: {config}")
        else:
            logger.info("No .meta file found or failed to load")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return config, meta_path, device

    @staticmethod
    def _load_and_prepare_model(model_path, config, meta_path, device, logger):
        """Load model state and prepare model instance."""
        from .serialisation import SerialisationService

        # Load state dict
        state_dict = SerialisationService.load_model_state(model_path, device, logger)

        # Infer parameters from state dict
        detected_params = ModelLoader._infer_params_from_state_dict(state_dict, config.vocab_size, logger)
        config.embed_dim = detected_params.get("embed_dim", config.embed_dim)
        config.ff_dim = detected_params.get("ff_dim", config.ff_dim)
        config.n_heads = detected_params.get("n_heads", config.n_heads)
        config.n_layers = detected_params.get("n_layers", config.n_layers)

        logger.info(
            f"Inferred parameters: embed_dim={config.embed_dim}, ff_dim={config.ff_dim}, "
            f"n_heads={config.n_heads}, n_layers={config.n_layers}"
        )
        logger.info(f"Loading model with config: {config}")

        # Validate config
        if not config.validate():
            logger.warning("Config validation failed, proceeding with loading")

        # Create and load model
        model = SmallTransformer(config)
        ModelLoader._load_state_dict_with_fallback(model, state_dict, logger)

        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model, config

    @staticmethod
    def _load_state_dict_with_fallback(model, state_dict, logger):
        """Load state dict with parameter mapping fallback."""
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if unexpected_keys:
            logger.info("Found parameter format differences, attempting conversion...")
            mapped_state_dict = ModelLoader._map_old_to_new_params(state_dict, model)
            if mapped_state_dict:
                missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
                if missing_keys or unexpected_keys:
                    logger.warning(
                        f"Still {len(missing_keys)} missing, {len(unexpected_keys)} unexpected after conversion"
                    )
            else:
                logger.warning("Parameter mapping failed - model may not work correctly")

        # Log remaining issues
        if missing_keys:
            logger.warning(f"{len(missing_keys)} keys missing in saved model")
        if unexpected_keys:
            logger.warning(f"{len(unexpected_keys)} unexpected keys in saved model")
        # Note: Missing keys indicate incomplete loading, but we allow it for compatibility

    @staticmethod
    def _handle_loading_error(error, model_path, config, device, logger):
        """Handle and log model loading errors with debugging info."""
        logger.error(f"Error loading model: {str(error)}")
        import traceback

        logger.debug("Full traceback:")
        logger.debug(traceback.format_exc())

        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Config: {config}")
        logger.debug(f"Device: {device}")

    @staticmethod
    def _infer_params_from_state_dict(state_dict, vocab_size, logger=None):
        """Infer model parameters from the saved state dictionary."""
        if logger is None:
            raise ValueError("logger parameter is required")
        params = {}

        # Infer embed_dim from embed.weight
        if "embed.weight" in state_dict:
            embed_weight = state_dict["embed.weight"]
            if len(embed_weight.shape) == 2:
                params["embed_dim"] = embed_weight.shape[1]
                logger.info(f"Detected embed_dim from saved model: {params['embed_dim']}")

        # Infer ff_dim from linear1.weight (feed-forward layer)
        embed_dim = params.get("embed_dim")
        if embed_dim:
            for key, param in state_dict.items():
                if "linear1.weight" in key and len(param.shape) == 2:
                    # linear1.weight shape is [ff_dim, embed_dim]
                    ff_dim = param.shape[0]
                    # Validate that ff_dim makes sense (should be larger than embed_dim)
                    if ff_dim >= embed_dim and ff_dim <= embed_dim * 8:  # Reasonable range
                        params["ff_dim"] = ff_dim
                        logger.info(f"Detected ff_dim from saved model: {params['ff_dim']}")
                    else:
                        logger.warning(f"Detected ff_dim={ff_dim} seems unreasonable for embed_dim={embed_dim}")
                    break

        # Infer n_heads from attention projection weights
        embed_dim = params.get("embed_dim")
        if embed_dim:
            # n_heads should be a reasonable divisor of embed_dim
            # Try common head counts in order of preference
            for possible_heads in [8, 12, 16, 4, 6, 10, 2, 14, 18, 20]:
                if embed_dim % possible_heads == 0:
                    head_dim = embed_dim // possible_heads
                    # Ensure reasonable head dimension (not too small or too large)
                    if 32 <= head_dim <= 256:
                        params["n_heads"] = possible_heads
                        logger.info(f"Detected n_heads from saved model: {params['n_heads']} (head_dim={head_dim})")
                        break

        # Count transformer layers
        transformer_keys = [k for k in state_dict.keys() if k.startswith("transformer.layers.")]
        layer_indices = set()
        for key in transformer_keys:
            parts = key.split(".")
            if len(parts) >= 3:
                try:
                    layer_idx = int(parts[2])
                    layer_indices.add(layer_idx)
                except ValueError:
                    pass
        params["n_layers"] = len(layer_indices)
        logger.info(f"Detected n_layers from saved model: {params['n_layers']}")

        # Validate inferred parameters
        embed_dim = params.get("embed_dim")
        n_heads = params.get("n_heads")
        ff_dim = params.get("ff_dim")
        n_layers = params.get("n_layers")

        if embed_dim and n_heads:
            head_dim = embed_dim // n_heads
            if head_dim < 16 or head_dim > 512:
                logger.warning(f"Inferred head_dim={head_dim} seems unreasonable, removing n_heads inference")
                params.pop("n_heads", None)

        if ff_dim and embed_dim and ff_dim < embed_dim:
            logger.warning(f"Inferred ff_dim={ff_dim} < embed_dim={embed_dim}, seems wrong")
            params.pop("ff_dim", None)

        return params

    @staticmethod
    def _load_meta_params(model_path, logger=None):
        """Load model parameters from a .meta file if it exists."""
        if logger is None:
            raise ValueError("logger parameter is required")
        # Extract model name from path for ModelPaths
        model_name = os.path.basename(model_path)
        if model_name.endswith(".pt"):
            model_name = model_name[:-3]

        meta_path = ModelPaths.get_meta_path(model_name)

        config = ModelConfig.load_from_file(meta_path, logger)
        if config:
            logger.info(f"Loaded config from {meta_path}: {config}")
        return config

    @staticmethod
    def _map_old_to_new_params(old_state_dict, model):
        """Map old PyTorch transformer parameter names to new custom attention parameter names."""
        new_state_dict = {}
        mapped_count = 0

        # Copy parameters that don't need mapping
        for key, param in old_state_dict.items():
            if not key.startswith("transformer.layers.") or "self_attn" not in key:
                new_state_dict[key] = param

        # Map transformer layer parameters
        for key, param in old_state_dict.items():
            if key.startswith("transformer.layers."):
                parts = key.split(".")
                if len(parts) >= 4:
                    layer_idx = parts[2]
                    param_path = ".".join(parts[3:])

                    # Map self-attention parameters from old format to new format
                    if param_path == "self_attn.in_proj_weight":
                        embed_dim = getattr(model, "embed_dim", 64)  # Default fallback
                        if param.shape[0] == 3 * embed_dim and param.shape[1] == embed_dim:
                            q_weight = param[:embed_dim]
                            k_weight = param[embed_dim : 2 * embed_dim]
                            v_weight = param[2 * embed_dim :]

                            new_state_dict[f"transformer.layers.{layer_idx}.self_attn.q_proj.weight"] = q_weight
                            new_state_dict[f"transformer.layers.{layer_idx}.self_attn.k_proj.weight"] = k_weight
                            new_state_dict[f"transformer.layers.{layer_idx}.self_attn.v_proj.weight"] = v_weight
                            mapped_count += 3

                    elif param_path == "self_attn.in_proj_bias":
                        embed_dim = getattr(model, "embed_dim", 64)
                        if param.shape[0] == 3 * embed_dim:
                            q_bias = param[:embed_dim]
                            k_bias = param[embed_dim : 2 * embed_dim]
                            v_bias = param[2 * embed_dim :]

                            new_state_dict[f"transformer.layers.{layer_idx}.self_attn.q_proj.bias"] = q_bias
                            new_state_dict[f"transformer.layers.{layer_idx}.self_attn.k_proj.bias"] = k_bias
                            new_state_dict[f"transformer.layers.{layer_idx}.self_attn.v_proj.bias"] = v_bias
                            mapped_count += 3

                    elif param_path == "self_attn.out_proj.weight":
                        new_state_dict[f"transformer.layers.{layer_idx}.self_attn.out_proj.weight"] = param
                        mapped_count += 1

                    elif param_path == "self_attn.out_proj.bias":
                        new_state_dict[f"transformer.layers.{layer_idx}.self_attn.out_proj.bias"] = param
                        mapped_count += 1

                    # Copy other layer parameters
                    elif param_path in [
                        "linear1.weight",
                        "linear1.bias",
                        "linear2.weight",
                        "linear2.bias",
                        "norm1.weight",
                        "norm1.bias",
                        "norm2.weight",
                        "norm2.bias",
                    ]:
                        new_state_dict[f"transformer.layers.{layer_idx}.{param_path}"] = param

        if mapped_count > 0:
            print(f"Converted {mapped_count} parameters from old format")
        return new_state_dict if new_state_dict else None

    @staticmethod
    def get_model_info(model_path, logger=None):
        """Get information about a saved model without loading it."""
        if logger is None:
            raise ValueError("logger parameter is required")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(model_path, map_location=device)

            # Try to infer model parameters from state_dict
            embed_weight = state_dict.get("embed.weight")
            if embed_weight is not None:
                vocab_size = embed_weight.shape[0]
                embed_dim = embed_weight.shape[1]
            else:
                vocab_size = embed_dim = None

            # Count transformer layers
            transformer_keys = [k for k in state_dict.keys() if k.startswith("transformer.layers.")]
            layer_indices = set()
            for key in transformer_keys:
                if "transformer.layers." in key:
                    parts = key.split(".")
                    if len(parts) >= 3:
                        try:
                            layer_idx = int(parts[2])
                            layer_indices.add(layer_idx)
                        except ValueError:
                            pass

            n_layers = len(layer_indices)

            return {
                "vocab_size": vocab_size,
                "embed_dim": embed_dim,
                "n_layers": n_layers,
                "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else None,
            }

        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None

    @staticmethod
    def save_model_with_meta(model, model_path, logger, config=None, meta_path=None):
        """Save a model along with its metadata for robust loading."""
        if config is None or not isinstance(config, ModelConfig):
            config = ModelConfig()

        # Update config with actual model parameters
        config.update_from_model(model)

        # Validate config before saving
        if not config.validate():
            logger.warning("Config validation failed, but proceeding with saving")

        if meta_path is None:
            # Handle .meta file path using ModelPaths
            model_name = os.path.basename(model_path)
            if model_name.endswith(".pt"):
                model_name = model_name[:-3]

            # Ensure model directory exists
            ModelPaths.ensure_model_dir(model_name)

            # Get proper paths using ModelPaths
            model_path = ModelPaths.get_model_path(model_name)
            meta_path = ModelPaths.get_meta_path(model_name)

        try:
            # Save the model state dict
            torch.save(model.state_dict(), model_path)

            # Save metadata using ConfigPersistence
            from ..model_config import ConfigPersistence

            ConfigPersistence.save_to_file(config, meta_path, logger)

            logger.info(f"Model saved to {model_path} with metadata at {meta_path}")
            logger.debug(f"Saved config: {config}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
