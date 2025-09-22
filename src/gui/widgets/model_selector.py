import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import pyqtSignal

from ..managers.model_manager import ModelManager
from ..styles import COMBO_STYLE, AVAILABLE_LABEL_STYLE
from ...core.model_discovery import ModelDiscovery


class ModelSelector(QWidget):
    """Model selection and loading interface"""

    model_changed = pyqtSignal(str)  # Emit model name when selection changes

    def __init__(self, model_manager: ModelManager):
        super().__init__()
        self.model_manager = model_manager
        self.setup_ui()

    def setup_ui(self):
        """Setup the model selection UI"""
        layout = QVBoxLayout()

        # Model selection combo box
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(COMBO_STYLE)
        self.model_combo.currentTextChanged.connect(self.load_selected_model)

        available_label = QLabel("Available Models:")
        available_label.setStyleSheet(AVAILABLE_LABEL_STYLE)
        layout.addWidget(available_label)
        layout.addWidget(self.model_combo)

        self.setLayout(layout)

        # Refresh model list
        self.refresh_model_list()

    def refresh_model_list(self):
        """Refresh the list of available models"""
        # Remember current selection
        current_selection = self.model_combo.currentText()

        self.model_combo.clear()
        available_models = ModelDiscovery.get_available_models()

        if available_models:
            # Create mapping from display names to model identifiers
            self.model_name_map = {}  # display_name -> model_identifier
            display_names = []

            for display_name, model_identifier in available_models:
                self.model_name_map[display_name] = model_identifier
                display_names.append(display_name)

            for display_name in display_names:
                self.model_combo.addItem(display_name)

            # Try to maintain previous selection if still available
            if current_selection in display_names:
                self.model_combo.setCurrentText(current_selection)
            elif display_names:
                # Default to first model if previous not available
                self.model_combo.setCurrentText(display_names[0])
        else:
            self.model_combo.addItem("No models available")
            self.model_name_map = {}

    def load_selected_model(self, display_name):
        """Load the selected model"""
        if not display_name or display_name == "No models available":
            return

        # Get the model identifier from the display name
        model_identifier = self.model_name_map.get(display_name, display_name)

        try:
            success = self.model_manager.load_model(model_identifier)
            # Emit signal to notify about model change
            self.model_changed.emit(display_name)
        except Exception as e:
            print(f"Failed to load model {model_identifier}: {e}")
            # Still emit signal even on failure
            self.model_changed.emit(display_name)

    def get_current_model_name(self):
        """Get the currently selected model name"""
        display_name = self.model_combo.currentText()
        # Return the actual filename for compatibility with existing code
        return self.model_name_map.get(display_name, display_name)
