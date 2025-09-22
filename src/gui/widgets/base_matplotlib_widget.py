from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib

matplotlib.use("QtAgg")


class BaseMatplotlibWidget(QWidget):
    """
    Base class for matplotlib-based widgets with common setup and functionality.
    Provides dark theme styling, canvas setup, and resize handling.
    """

    def __init__(self, figsize=(8, 6)):
        super().__init__()
        self.figsize = figsize
        self.setup_matplotlib()
        self.setup_ui()

    def setup_matplotlib(self):
        """Setup matplotlib with dark theme"""
        plt.style.use("dark_background")
        plt.rcParams["axes.facecolor"] = "#2b2b2b"
        plt.rcParams["figure.facecolor"] = "#2b2b2b"

        self.figure, self.ax = plt.subplots(figsize=self.figsize)
        self.figure.patch.set_facecolor("#2b2b2b")
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="white", labelsize=10)
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")

        # Setup canvas
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(400, 300)

    def setup_ui(self):
        """Setup the UI layout - can be overridden by subclasses"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create a container to center the canvas
        canvas_container = QWidget()
        canvas_layout = QHBoxLayout()
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addStretch()  # Left spacer
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addStretch()  # Right spacer
        canvas_container.setLayout(canvas_layout)
        layout.addWidget(canvas_container)

    def apply_dark_theme(self):
        """Apply consistent dark theme styling to the current axes"""
        self.ax.set_facecolor("#1e1e1e")
        self.figure.set_facecolor("#2b2b2b")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")
        for spine in self.ax.spines.values():
            spine.set_color("white")

    def resizeEvent(self, a0):
        """Handle resize events to adjust figure size dynamically"""
        super().resizeEvent(a0)

        # Adjust figure size based on widget size
        width = self.width()
        height = self.height()

        # Calculate appropriate figure size (in inches) based on widget size
        fig_width = max(4, width / 100)
        fig_height = max(3, height / 100)

        # Update figure size
        self.figure.set_size_inches(fig_width, fig_height)

        # Redraw if there is data
        if hasattr(self, "has_data") and self.has_data:
            self.canvas.draw()

    def redraw(self):
        """Redraw the canvas - can be called by subclasses"""
        self.canvas.draw()

    def clear_axes(self):
        """Clear the axes - utility method for subclasses"""
        self.ax.clear()
        self.apply_dark_theme()
