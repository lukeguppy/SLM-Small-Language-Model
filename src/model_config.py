from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import os


@dataclass
class ModelConfig:
    """Configuration class for model parameters and training settings"""

    # Core model parameters
    vocab_size: int = 1000  # This will be set dynamically from vocab file
    embed_dim: int = 512
    ff_dim: int = 1024
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.3

    # Training parameters
    epochs: int = 50
    lr: float = 0.0005
    weight_decay: float = 0.0001
    batch_size: int = 64

    # Data parameters
    train_size: int = 50000
    val_size: int = 10000
    test_size: int = 10000
    vocab_path: str = ""  # Will be set dynamically or from DataPaths

    # Model architecture parameters
    max_seq_len: int = 2048
    model_name: str = "model"

    # Runtime metadata (set when model is saved/loaded)
    saved_date: Optional[str] = None
    saved_time: Optional[str] = None
    model_path: Optional[str] = None

    def __post_init__(self):
        """Set runtime metadata if not provided"""
        if self.saved_date is None:
            now = datetime.now()
            self.saved_date = now.strftime("%Y-%m-%d")
            self.saved_time = now.strftime("%H:%M")

        # Set default vocab_path if not provided
        if not self.vocab_path:
            from .core.paths import DataPaths

            self.vocab_path = DataPaths.get_vocab_path()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialisation"""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "vocab_path": self.vocab_path,
            "max_seq_len": self.max_seq_len,
            "model_name": self.model_name,
            "saved_date": self.saved_date,
            "saved_time": self.saved_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary"""
        # Create instance with provided data
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                # Special handling for dropout if it's saved as Dropout object string
                if key == "dropout" and isinstance(value, str) and "Dropout" in value:
                    try:
                        p_str = value.split("p=")[1].split(",")[0].strip()
                        value = float(p_str)
                    except:
                        pass  # Keep original value if parsing fails
                # For Optional fields, allow None values to override defaults
                if key in ("saved_date", "saved_time", "model_path") or value is not None:
                    setattr(config, key, value)  # type: ignore
        return config

    def save_to_file(self, filepath: str, logger) -> bool:
        """Save config to key-value format file."""
        return ConfigPersistence.save_to_file(self, filepath, logger)

    @classmethod
    def load_from_file(cls, filepath: str, logger) -> Optional["ModelConfig"]:
        """Load config from key-value format file."""
        return ConfigPersistence.load_from_file(filepath, logger)

    def update_from_model(self, model):
        """Update config with actual model parameters"""
        if hasattr(model, "embed_dim"):
            self.embed_dim = model.embed_dim
        if hasattr(model, "n_heads"):
            self.n_heads = model.n_heads
        if hasattr(model, "n_layers"):
            self.n_layers = model.n_layers

        # Try to infer ff_dim from model structure
        if hasattr(model, "ff_dim"):
            self.ff_dim = model.ff_dim
        else:
            # Try to infer from linear layers
            for name, module in model.named_modules():
                if "linear1" in name and hasattr(module, "out_features"):
                    self.ff_dim = module.out_features
                    break

        # Try to infer dropout
        if hasattr(model, "dropout"):
            # Handle both Dropout layer and float values
            if hasattr(model.dropout, "p"):
                self.dropout = model.dropout.p
            else:
                # Assume it's already a float value
                self.dropout = float(model.dropout)
        else:
            # Try to infer from dropout layers
            dropout_values = []
            for name, module in model.named_modules():
                if hasattr(module, "p"):
                    dropout_values.append(module.p)
            if dropout_values:
                self.dropout = max(set(dropout_values), key=dropout_values.count)

    def validate(self) -> bool:
        """Validate that config parameters are consistent and reasonable."""
        # Validate core model parameters
        if self.vocab_size <= 0:
            print(f"Error: vocab_size must be positive, got {self.vocab_size}")
            return False

        if self.embed_dim <= 0:
            print(f"Error: embed_dim must be positive, got {self.embed_dim}")
            return False

        if self.n_heads <= 0:
            print(f"Error: n_heads must be positive, got {self.n_heads}")
            return False

        if self.ff_dim <= 0:
            print(f"Error: ff_dim must be positive, got {self.ff_dim}")
            return False

        if self.n_layers <= 0:
            print(f"Error: n_layers must be positive, got {self.n_layers}")
            return False

        # Validate parameter relationships
        if self.embed_dim % self.n_heads != 0:
            print(f"Error: embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})")
            return False

        # Validate reasonable ranges
        if self.n_heads > self.embed_dim:
            print(
                f"Warning: n_heads ({self.n_heads}) should typically be less than or equal to embed_dim ({self.embed_dim})"
            )

        if self.ff_dim < self.embed_dim:
            print(f"Warning: ff_dim ({self.ff_dim}) is typically larger than embed_dim ({self.embed_dim})")

        return True

    def __str__(self) -> str:
        """String representation of config"""
        return f"ModelConfig(vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, n_heads={self.n_heads}, n_layers={self.n_layers}, dropout={self.dropout})"


class ConfigPersistence:
    """
    Handles persistence operations for ModelConfig.

    This class separates file I/O operations from the ModelConfig data class,
    following the Single Responsibility Principle.
    """

    @staticmethod
    def save_to_file(config: ModelConfig, filepath: str, logger) -> bool:
        """Save config to key-value format file."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for key, value in config.to_dict().items():
                    f.write(f"{key}: {value}\n")
            logger.debug(f"Config saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {e}")
            return False

    @staticmethod
    def load_from_file(filepath: str, logger) -> Optional[ModelConfig]:
        """Load config from key-value format file."""

        if not os.path.exists(filepath):
            logger.debug(f"Config file not found: {filepath}")
            return None

        data = {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue  # Skip empty lines and comments

                    if ":" not in line:
                        logger.warning(f"Skipping malformed line {line_num} in {filepath}: {line}")
                        continue

                    try:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()

                        # Handle None/empty values for Optional fields
                        if value in ("", "None", "null", "N/A"):
                            data[key] = None
                        else:
                            # Try to convert to appropriate type
                            data[key] = ConfigPersistence._parse_value(value)
                    except ValueError as e:
                        logger.warning(f"Failed to parse line {line_num} in {filepath}: {e}")
                        continue

            config = ModelConfig.from_dict(data)
            logger.debug(f"Config loaded from {filepath}")
            return config

        except Exception as e:
            logger.error(f"Error loading config from {filepath}: {e}")
            return None

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse a string value into the appropriate Python type."""
        # Try to convert to appropriate type
        if "." not in value:
            # Try int first
            try:
                return int(value)
            except ValueError:
                pass
        else:
            # Try float
            try:
                return float(value)
            except ValueError:
                pass

        # Handle boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Return as string if no other type matches
        return value

    @staticmethod
    def save_config_with_backup(config: ModelConfig, filepath: str, logger) -> bool:
        """Save config with automatic backup of existing file."""

        # Create backup if file exists
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            try:
                import shutil

                shutil.copy2(filepath, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        return ConfigPersistence.save_to_file(config, filepath, logger)
