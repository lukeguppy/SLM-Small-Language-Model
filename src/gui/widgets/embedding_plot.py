from PyQt5.QtWidgets import QWidget, QVBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ...core.embedding_utils import reduce_embeddings_2d
from .base_matplotlib_widget import BaseMatplotlibWidget


class EmbeddingPlotWidget(BaseMatplotlibWidget):
    def __init__(self):
        super().__init__(figsize=(8, 6))

        # Store data for hover functionality
        self.reduced_embeddings = None
        self.tokens = None
        self.hover_annotation = None
        self.has_data = False  # For resize handling

        # Connect hover event
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def update(self, embeddings, method="tsne", tokens=None):
        self.ax.clear()

        # Remove existing colorbars
        colorbar_axes = [ax for ax in self.figure.axes if ax != self.ax]
        for cb_ax in colorbar_axes:
            cb_ax.remove()

        if embeddings is not None and len(embeddings) > 0:
            try:
                # Handle 3D tensor (batch_size, seq_len, embed_dim) -> 2D (seq_len, embed_dim)
                if hasattr(embeddings, "shape") and len(embeddings.shape) == 3:
                    embeddings = embeddings.squeeze(0)  # Remove batch dimension
                reduced = reduce_embeddings_2d(embeddings, method)

                if reduced is not None:
                    # Store data for hover functionality
                    self.reduced_embeddings = reduced
                    self.tokens = tokens or [f"Token {i}" for i in range(len(reduced))]

                    scatter = self.ax.scatter(
                        reduced[:, 0],
                        reduced[:, 1],
                        c=range(len(reduced)),
                        cmap="viridis",
                        s=100,
                        alpha=0.7,
                        edgecolors="white",
                        linewidth=1,
                    )
                    self.ax.autoscale()

                    # Add colorbar
                    cbar = self.figure.colorbar(scatter, ax=self.ax, shrink=0.8)
                    cbar.set_label("Token Position", color="white", fontsize=10)
                    cbar.ax.tick_params(colors="white", labelsize=8)

                    self.ax.set_xlabel("Dimension 1", color="white", fontsize=11)
                    self.ax.set_ylabel("Dimension 2", color="white", fontsize=11)
                    self.ax.set_title(f"Embeddings ({method.upper()})", color="white", fontsize=13, pad=20)
                    self.ax.grid(True, alpha=0.3, color="white")
                else:
                    self.reduced_embeddings = None
                    self.tokens = tokens or []
                    self.ax.text(
                        0.5,
                        0.5,
                        "Failed to reduce embeddings",
                        ha="center",
                        va="center",
                        transform=self.ax.transAxes,
                        color="red",
                        fontsize=12,
                    )
            except Exception as e:
                self.ax.text(
                    0.5,
                    0.5,
                    f"Error reducing embeddings:\n{str(e)}",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    color="red",
                    fontsize=12,
                )
        else:
            self.ax.text(
                0.5,
                0.5,
                "No embedding data\nRun analysis first",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                color="white",
                fontsize=14,
            )
            self.ax.set_title("Embeddings Plot", color="white", fontsize=13)

        self.canvas.draw()

    def resizeEvent(self, a0):
        """Handle resize events to adjust figure size dynamically"""
        super().resizeEvent(a0)

        # Adjust figure size based on widget size
        width = self.width()
        height = self.height()

        # Calculate figure size (inches) based on widget size
        # Use 100 pixels per inch as DPI
        fig_width = min(8, max(4, width / 100))
        fig_height = min(6, max(3, height / 100))

        # Update figure size
        self.figure.set_size_inches(fig_width, fig_height)

        # Redraw if there is data
        if self.reduced_embeddings is not None:
            self.has_data = self.reduced_embeddings is not None
            self.canvas.draw()

    def on_hover(self, event):
        """Handle mouse hover events to show token information"""
        if event.inaxes != self.ax or self.reduced_embeddings is None:
            return

        if self.hover_annotation:
            self.hover_annotation.remove()
            self.hover_annotation = None

        # Find closest point
        if self.reduced_embeddings is not None and len(self.reduced_embeddings) > 0:
            distances = (self.reduced_embeddings[:, 0] - event.xdata) ** 2 + (
                self.reduced_embeddings[:, 1] - event.ydata
            ) ** 2
            closest_idx = distances.argmin()
            closest_distance = distances[closest_idx] ** 0.5

            # Only show annotation if mouse is close enough to a point
            if closest_distance < 5.0 and self.reduced_embeddings is not None:  # Threshold for hover distance
                token_text = (
                    self.tokens[closest_idx]
                    if self.tokens and closest_idx < len(self.tokens)
                    else f"Token {closest_idx}"
                )
                self.hover_annotation = self.ax.annotate(
                    token_text,
                    xy=(self.reduced_embeddings[closest_idx, 0], self.reduced_embeddings[closest_idx, 1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    fontsize=10,
                    color="black",
                )

        self.canvas.draw()
