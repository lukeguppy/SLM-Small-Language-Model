from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams["axes.facecolor"] = "#2b2b2b"
plt.rcParams["figure.facecolor"] = "#2b2b2b"


class LossPlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.train_losses = []
        self.val_losses = []
        self.batch_counts = []
        self.epochs = []
        self.epoch_batch_positions = []
        self.visible = False
        self.hide()

    def update_plot_batch(self, batch_count, train_loss, val_loss=None):
        """Update plot with batch-level data"""
        if batch_count >= 100:  # Only show after 100 batches
            if not self.visible:
                self.show()
                self.visible = True

            self.batch_counts.append(batch_count)
            self.train_losses.append(train_loss)

            # Only add validation loss if provided (at epoch boundaries)
            if val_loss is not None:
                self.val_losses.append(val_loss)

            self._redraw_plot()

    def update_plot_epoch(self, epoch, train_loss, val_loss):
        """Update plot with epoch-level data (for validation loss)"""
        if epoch >= 1:
            if not self.visible:
                self.show()
                self.visible = True

            self.epochs.append(epoch)
            # Add validation loss at epoch boundary
            if len(self.val_losses) < len(self.epochs):
                self.val_losses.append(val_loss)
            # Record the batch position for this epoch
            if self.batch_counts:
                self.epoch_batch_positions.append(self.batch_counts[-1])

            self._redraw_plot()

    def _redraw_plot(self):
        """Redraw the plot with current data"""
        self.ax.clear()

        if self.batch_counts:
            # Plot training loss vs batch count
            self.ax.plot(self.batch_counts, self.train_losses, "b-", label="Train Loss", linewidth=2, alpha=0.7)

        if self.val_losses and self.epochs and self.epoch_batch_positions:
            # Plot validation loss at the recorded batch positions
            self.ax.plot(self.epoch_batch_positions, self.val_losses, "r-", label="Val Loss", linewidth=2, marker="o")

        # Set x-axis label
        self.ax.set_xlabel("Batches")
        self.ax.set_ylabel("Loss")
        # Remove title to save space and improve readability
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        # Match GUI group box background
        self.ax.set_facecolor("#2b2b2b")
        self.figure.set_facecolor("#2b2b2b")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")
        for spine in self.ax.spines.values():
            spine.set_color("white")

        self.figure.tight_layout()
        self.canvas.draw()

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.batch_counts = []
        self.epochs = []
        self.epoch_batch_positions = []
        self.visible = False
        self.hide()
        self.ax.clear()
        self.canvas.draw()
