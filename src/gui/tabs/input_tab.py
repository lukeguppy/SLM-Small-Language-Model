from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
)
from PyQt5.QtCore import pyqtSignal
from typing import cast, Any
from ..styles import GROUP_STYLE, BUTTON_STYLE, RESULTS_LABEL_STYLE, MAIN_WINDOW_STYLE
from ..widgets.autocomplete import AutocompleteTextEdit
from ..widgets.model_selector import ModelSelector


class InputTab(QWidget):
    """Input and prediction interface tab"""

    # Signals
    attention_requested = pyqtSignal(str)  # Emit text for attention analysis

    def __init__(self, model_manager, vocab_manager):
        super().__init__()
        self.model_manager = model_manager
        self.vocab_manager = vocab_manager
        self.setStyleSheet(MAIN_WINDOW_STYLE)
        self.setup_ui()

    def setup_ui(self):
        """Setup the input tab UI"""
        layout = QVBoxLayout()

        # Model selection at top
        model_select_group = QGroupBox("Model Selection")
        model_select_group.setStyleSheet(GROUP_STYLE)
        model_select_layout = QHBoxLayout()

        self.model_selector = ModelSelector(self.model_manager)
        self.model_selector.model_changed.connect(self.on_model_changed)
        model_select_layout.addWidget(self.model_selector)
        model_select_layout.addStretch()

        model_select_group.setLayout(model_select_layout)
        layout.addWidget(model_select_group)

        # Input section
        input_group = QGroupBox("Input Sentence")
        input_group.setStyleSheet(GROUP_STYLE)
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(10, 0, 10, 0)
        input_layout.setSpacing(0)

        # Create autocomplete text input
        self.text_input = AutocompleteTextEdit(self.vocab_manager, self.model_manager)
        self.text_input.prediction_requested.connect(self.update_predictions)
        input_layout.addWidget(self.text_input)

        # Separator line
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #4c4c4c; margin: 12px 0 8px 0;")
        input_layout.addWidget(separator)

        # Predictions section header
        predictions_header = QLabel("Next Token Predictions")
        predictions_header.setStyleSheet(
            """
            color: #c0c0c0;
            font-size: 14px;
            font-weight: bold;
            padding: 4px 0;
            margin-bottom: 4px;
        """
        )
        input_layout.addWidget(predictions_header)

        # Predictions content
        self.next_token_label = QLabel("Start typing to see next token predictions...")
        self.next_token_label.setStyleSheet(
            """
            color: #c0c0c0;
            font-size: 14px;
            padding: 8px 12px;
            background-color: #1e1e1e;
            border-radius: 4px;
            margin: 0 0 8px 0;
        """
        )
        input_layout.addWidget(self.next_token_label)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Attention Matrix Control
        attention_group = QGroupBox("Attention Matrix Control")
        attention_group.setStyleSheet(GROUP_STYLE)
        attention_layout = QVBoxLayout()

        # Run button - now prominently placed with the layer controls
        self.run_button = QPushButton("Show Attention Matrix")
        self.run_button.clicked.connect(self.request_attention_analysis)
        self.run_button.setStyleSheet(BUTTON_STYLE)
        attention_layout.addWidget(self.run_button)

        attention_group.setLayout(attention_layout)
        layout.addWidget(attention_group)

        # Results section for feedback
        results_group = QGroupBox("Status")
        results_group.setStyleSheet(GROUP_STYLE)
        results_layout = QVBoxLayout()

        self.feedback_label = QLabel("Ready")
        self.feedback_label.setStyleSheet(RESULTS_LABEL_STYLE)
        results_layout.addWidget(self.feedback_label)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Show feedback for initial model loading
        self.show_initial_model_feedback()

        layout.addStretch()
        self.setLayout(layout)

    def update_predictions(self):
        """Update next token predictions based on current input"""
        text = self.text_input.toPlainText().strip()
        if not text:
            self.next_token_label.setText(
                "Start typing to see next token predictions..."
            )
            return

        predictions = self.model_manager.predict_next_tokens(text, top_k=3)
        if predictions:
            prediction_text = (
                f"Next tokens: {', '.join([token for token, _ in predictions])}"
            )
            self.next_token_label.setText(prediction_text)
        else:
            self.next_token_label.setText("Unable to predict next token")

    def on_model_changed(self, model_name):
        """Handle model selection change and provide feedback"""
        if not model_name or model_name == "No models available":
            return

        # Clear previous predictions
        self.feedback_label.setText("Loading model...")

        # Check if the model was loaded successfully by checking if it exists
        if self.model_manager.get_model_service().get_model() is not None:
            self.feedback_label.setText(f"Model '{model_name}' loaded successfully!")
        else:
            self.feedback_label.setText(f"Failed to load model '{model_name}'")

    def request_attention_analysis(self):
        """Request attention matrix analysis for current input"""
        text = self.text_input.toPlainText().strip()
        if not text:
            self.feedback_label.setText("Please enter a sentence to analyse")
            return

        cast(Any, self.attention_requested).emit(text)

    def show_initial_model_feedback(self):
        """Show feedback for the initially loaded model"""
        current_model = self.model_selector.get_current_model_name()
        if current_model and current_model != "No models available":
            if self.model_manager.get_model_service().get_model() is not None:
                self.feedback_label.setText(
                    f"Model '{current_model}' loaded successfully!"
                )
            else:
                self.feedback_label.setText(f"Failed to load model '{current_model}'")
