from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QProgressBar
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams["axes.facecolor"] = "#2b2b2b"
plt.rcParams["figure.facecolor"] = "#2b2b2b"

from ..model_config import ModelConfig
from ..core.logger import create_logger
from .managers.vocab_manager import VocabManager
from .managers.model_manager import ModelManager
from .managers.training_manager import TrainingManager


class CustomProgressBar(QProgressBar):
    def text(self):
        if self.maximum() == 0:
            return "0.00%"
        percentage = max(0, self.value() / self.maximum() * 100)
        return "%.2f%%" % percentage


class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SLM Attention Explorer")
        self.setMinimumSize(1200, 800)
        self.showMaximized()

        # Initialise config with ModelConfig defaults, only override GUI-specific values
        self.config = ModelConfig()

        # Create logger instance
        self.logger = create_logger()

        # Initialise managers with logger injection
        self.vocab_manager = VocabManager(self.logger.get_gui_logger())
        self.model_manager = ModelManager(self.vocab_manager, self.config, self.logger.get_gui_logger())
        self.training_manager = TrainingManager(self.model_manager, self.logger.get_training_logger())

        # Load vocabulary
        self.vocab_manager.load_vocabulary(self.config.vocab_path)
        self.vocab_size = self.vocab_manager.get_vocab_size()
        self.config.vocab_size = self.vocab_size  # Update config with actual vocab size
        self.vocab = self.vocab_manager.get_vocab()
        self.id_to_token = self.vocab_manager.get_id_to_token()

        # Data files are loaded directly from text files
        self.embeddings = None
        self.current_token_ids = None

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        from .tabs.input_tab import InputTab

        self.input_tab = InputTab(self.model_manager, self.vocab_manager)
        self.input_tab.attention_requested.connect(self.switch_to_visualisation)
        self.tab_widget.addTab(self.input_tab, "Input")

        from .tabs.training_tab import TrainingTab

        self.training_tab = TrainingTab(self.training_manager, self.model_manager, self.config)
        self.training_manager.epoch_completed.connect(self.training_tab.update_training_status)
        self.training_manager.batch_completed.connect(self.training_tab.update_batch_progress)
        self.training_manager.batch_loss_updated.connect(self.training_tab.loss_plot_widget.update_plot_batch)
        self.training_manager.progress_updated.connect(self.training_tab.update_training_status_epoch)
        self.training_manager.training_finished.connect(self.training_tab.on_training_finished)
        self.training_manager.training_finished.connect(self.input_tab.model_selector.refresh_model_list)
        self.training_manager.time_remaining_updated.connect(self.training_tab.update_time_remaining)
        self.tab_widget.addTab(self.training_tab, "Training")

        from .tabs.visualisation_tab import VisualisationTab

        self.visualisation_tab = VisualisationTab()
        self.tab_widget.addTab(self.visualisation_tab, "Visualisation")

        # Connect tab change signal to hide autocomplete overlays
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def switch_to_visualisation(self, text):
        """Switch to visualisation tab and update with attention analysis for given text"""
        self.tab_widget.setCurrentIndex(2)

        # Perform attention analysis
        attention_weights = self.model_manager.get_attention_weights(text)
        embeddings = self.model_manager.get_embeddings(text)
        tokens = self.vocab_manager.text_to_tokens(text)

        # Update attention matrix if there is data
        if attention_weights is not None and tokens:
            # Process attention weights for all layers
            all_layer_matrices = []
            token_texts = [self.vocab_manager.get_id_to_token().get(token_id, f"<{token_id}>") for token_id in tokens]

            if isinstance(attention_weights, list):
                # Multiple layers stored as list
                for layer_weights in attention_weights:
                    if layer_weights is not None:
                        # Average across heads: [n_heads, seq_len, seq_len] -> [seq_len, seq_len]
                        layer_matrix = layer_weights.squeeze(0).mean(dim=0).cpu().numpy()
                        all_layer_matrices.append(layer_matrix)
                    else:
                        # Fallback for missing layer
                        import numpy as np

                        all_layer_matrices.append(np.random.rand(len(tokens), len(tokens)))
            else:
                # Single layer or different format
                layer_matrix = attention_weights.squeeze(0).mean(dim=0).cpu().numpy()
                all_layer_matrices.append(layer_matrix)

            # Update attention matrix widget
            self.visualisation_tab.update_attention_matrices(all_layer_matrices, token_texts)

        # Update embeddings if we have data
        if embeddings is not None and tokens:
            token_texts = [self.vocab_manager.get_id_to_token().get(token_id, f"<{token_id}>") for token_id in tokens]
            # Update embedding plot
            self.visualisation_tab.update_embeddings(embeddings, token_texts)

    def on_tab_changed(self, index):
        """Handle tab change events, hide autocomplete overlays when leaving input tab"""
        # Hide autocomplete overlays when switching away from input tab
        if hasattr(self, "input_tab") and self.tab_widget.currentIndex() != 0:
            self.input_tab.text_input.hide_all_overlays()


if __name__ == "__main__":
    app = QApplication([])
    window = MainGUI()
    window.show()
    app.exec_()
