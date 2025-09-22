from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit
from PyQt5.QtGui import QCursor
from ..styles import INPUT_STYLE, INFO_BUTTON_STYLE
from ...model_config import ModelConfig


class ImmediateTooltipLabel(QLabel):
    """A QLabel that shows tooltips immediately on hover"""

    def __init__(self, text="", tooltip_text="", parent=None):
        super().__init__(text, parent)
        self.tooltip_text = tooltip_text
        self.setMouseTracking(True)

    def enterEvent(self, a0):
        """Show tooltip immediately when mouse enters"""
        if self.tooltip_text:
            from PyQt5.QtWidgets import QToolTip

            QToolTip.showText(QCursor.pos(), self.tooltip_text, self)
        super().enterEvent(a0)

    def leaveEvent(self, a0):
        """Hide tooltip when mouse leaves"""
        from PyQt5.QtWidgets import QToolTip

        QToolTip.hideText()
        super().leaveEvent(a0)


class TrainingControls(QWidget):
    """Training parameter input and validation"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.parameter_inputs = {}
        self.setup_parameter_inputs()

    def setup_parameter_inputs(self):
        """Create all parameter input fields in two columns"""
        main_layout = QHBoxLayout()

        # Left column - Model Parameters
        left_layout = QVBoxLayout()
        left_layout.addLayout(
            self.create_parameter_input(
                "Embedding Dim:",
                self.config.embed_dim,
                "Size of token representations. Larger values capture more complex patterns but use more memory and compute.",
                "embed_dim",
            )
        )

        left_layout.addLayout(
            self.create_parameter_input(
                "FF Dim:",
                self.config.ff_dim,
                "Feed-forward dimension in transformer layers. Controls expansion size for processing.",
                "ff_dim",
            )
        )

        left_layout.addLayout(
            self.create_parameter_input(
                "Num Heads:",
                self.config.n_heads,
                "Number of attention heads for parallel processing. More heads can capture diverse relationships.",
                "n_heads",
            )
        )

        left_layout.addLayout(
            self.create_parameter_input(
                "Num Layers:",
                self.config.n_layers,
                "Depth of the transformer network. More layers learn complex patterns but increase training time.",
                "n_layers",
            )
        )

        left_layout.addLayout(
            self.create_parameter_input(
                "Dropout:",
                self.config.dropout,
                "Fraction of connections randomly disabled during training to prevent overfitting.",
                "dropout",
            )
        )

        left_layout.addLayout(
            self.create_parameter_input(
                "Weight Decay:",
                self.config.weight_decay,
                "Penalty on large weights to prevent overfitting and improve generalisation.",
                "weight_decay",
            )
        )

        # Right column - Training Parameters
        right_layout = QVBoxLayout()
        right_layout.addLayout(
            self.create_parameter_input(
                "Epochs:",
                self.config.epochs,
                "Number of complete passes through the training data. More epochs can improve learning but may overfit.",
                "epochs",
            )
        )

        right_layout.addLayout(
            self.create_parameter_input(
                "Learning Rate:",
                self.config.lr,
                "How much the model adjusts with each update. Higher values learn faster but risk overshooting the optimal solution.",
                "lr",
            )
        )

        right_layout.addLayout(
            self.create_parameter_input(
                "Train Size:", self.config.train_size, "Number of sentences used for training.", "train_size"
            )
        )

        right_layout.addLayout(
            self.create_parameter_input(
                "Val Size:", self.config.val_size, "Number of sentences used for validation.", "val_size"
            )
        )

        right_layout.addLayout(
            self.create_parameter_input(
                "Test Size:", self.config.test_size, "Number of sentences used for testing.", "test_size"
            )
        )

        right_layout.addLayout(
            self.create_parameter_input(
                "Batch Size:",
                self.config.batch_size,
                "Number of samples processed together. Larger batches are faster but require more memory.",
                "batch_size",
            )
        )

        right_layout.addLayout(
            self.create_parameter_input(
                "Vocab Path:",
                self.config.vocab_path,
                "Path to vocabulary file (relative to slm folder). Default: data/tokens.txt",
                "vocab_path",
            )
        )

        # Model Name (spans both columns)
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Model Name:"))
        from datetime import datetime

        default_name = f"model-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText(default_name)
        self.model_name_input.setStyleSheet(INPUT_STYLE)
        name_layout.addWidget(self.model_name_input)
        name_info = ImmediateTooltipLabel(
            "ⓘ", "Name for saving the trained model. Leave blank to use the default date-based name."
        )
        name_info.setStyleSheet(INFO_BUTTON_STYLE)
        name_layout.addWidget(name_info)
        name_layout.addStretch()

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Add model name at the bottom spanning both columns
        full_layout = QVBoxLayout()
        full_layout.addLayout(main_layout)
        full_layout.addLayout(name_layout)

        self.setLayout(full_layout)

    def create_parameter_input(self, label_text, config_value, tooltip_text, attribute_name):
        """Helper function to create a parameter input field with label and info button"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))

        input_field = QLineEdit(str(config_value))
        input_field.setStyleSheet(INPUT_STYLE)
        self.parameter_inputs[attribute_name] = input_field
        layout.addWidget(input_field)

        info_button = ImmediateTooltipLabel("ⓘ", tooltip_text)
        info_button.setStyleSheet(INFO_BUTTON_STYLE)
        layout.addWidget(info_button)

        layout.addStretch()
        return layout

    def get_training_parameters(self):
        """Get all training parameters as ModelConfig object"""
        params = {}
        for name, input_field in self.parameter_inputs.items():
            value = input_field.text().strip()
            if value:
                if name in [
                    "epochs",
                    "train_size",
                    "val_size",
                    "test_size",
                    "embed_dim",
                    "ff_dim",
                    "n_heads",
                    "n_layers",
                    "batch_size",
                ]:
                    params[name] = int(value)
                elif name in ["lr", "dropout", "weight_decay"]:
                    params[name] = float(value)
                else:
                    params[name] = value

        # Handle model name separately
        model_name = self.model_name_input.text().strip()
        if model_name:
            params["model_name"] = model_name
        else:
            params["model_name"] = self.model_name_input.placeholderText()

        # Create ModelConfig from params
        config = ModelConfig(
            vocab_size=self.config.vocab_size,
            embed_dim=params.get("embed_dim", self.config.embed_dim),
            n_heads=params.get("n_heads", self.config.n_heads),
            ff_dim=params.get("ff_dim", self.config.ff_dim),
            n_layers=params.get("n_layers", self.config.n_layers),
            dropout=params.get("dropout", self.config.dropout),
            epochs=params.get("epochs", self.config.epochs),
            lr=params.get("lr", self.config.lr),
            weight_decay=params.get("weight_decay", self.config.weight_decay),
            batch_size=params.get("batch_size", self.config.batch_size),
            train_size=params.get("train_size", self.config.train_size),
            val_size=params.get("val_size", self.config.val_size),
            test_size=params.get("test_size", self.config.test_size),
            vocab_path=params.get("vocab_path", self.config.vocab_path),
            model_name=params.get("model_name", self.config.model_name),
        )
        return config

    def set_parameters_enabled(self, enabled):
        """Enable or disable all parameter inputs"""
        for input_field in self.parameter_inputs.values():
            input_field.setEnabled(enabled)
        self.model_name_input.setEnabled(enabled)
