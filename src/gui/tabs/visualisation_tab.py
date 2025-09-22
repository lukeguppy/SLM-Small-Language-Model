from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from ..widgets.attention_matrix import AttentionMatrixWidget
from ..widgets.embedding_plot import EmbeddingPlotWidget
from ..styles import GROUP_STYLE, MAIN_WINDOW_STYLE


class VisualisationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(MAIN_WINDOW_STYLE)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Attention visualisation
        att_group = QGroupBox("Attention Matrix")
        att_group.setStyleSheet(GROUP_STYLE)
        att_layout = QVBoxLayout()

        self.att_matrix_widget = AttentionMatrixWidget()
        att_layout.addWidget(self.att_matrix_widget, stretch=1)

        att_group.setLayout(att_layout)
        layout.addWidget(att_group, stretch=1)

        # Embedding visualisation
        emb_group = QGroupBox("Embedding Plot")
        emb_group.setStyleSheet(GROUP_STYLE)
        emb_layout = QVBoxLayout()

        self.embedding_plot_widget = EmbeddingPlotWidget()
        emb_layout.addWidget(self.embedding_plot_widget, stretch=1)

        emb_group.setLayout(emb_layout)
        layout.addWidget(emb_group, stretch=1)

        self.setLayout(layout)

    def update_attention_matrices(self, matrices, tokens):
        self.att_matrix_widget.update_all_layers(matrices, tokens)

    def update_embeddings(self, embeddings, tokens):
        self.embedding_plot_widget.update(embeddings, tokens=tokens)
