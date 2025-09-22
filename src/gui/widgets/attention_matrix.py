from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QSlider, QHBoxLayout
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .base_matplotlib_widget import BaseMatplotlibWidget


class AttentionMatrixWidget(BaseMatplotlibWidget):
    def __init__(self):
        # Initialise with specific figsize
        super().__init__(figsize=(7, 4))

        # Custom layout setup
        layout = self.layout()
        assert layout is not None
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        layout.setSpacing(5)  # Reduce spacing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Ensure widget has dark background
        self.setStyleSheet("background-color: #2b2b2b;")

        # Adjust figure layout for labels and colorbar
        self.figure.subplots_adjust(bottom=0.15, left=0.12, right=0.88, top=0.85)

        # Constrain canvas size
        self.canvas.setMaximumSize(800, 600)  # Add maximum size constraint
        # Ensure canvas background matches the dark theme
        self.canvas.setStyleSheet("background-color: #2b2b2b;")

        # Create centered container for layer controls
        controls_container = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addStretch()  # Left spacer

        # Layer control slider
        self.layer_control_layout = QHBoxLayout()
        self.layer_label = QLabel("Layer:")
        self.layer_label.setStyleSheet("color: #c0c0c0; font-weight: bold;")
        self.layer_control_layout.addWidget(self.layer_label)

        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(0)  # Will be updated when data is loaded
        self.layer_slider.setValue(0)
        self.layer_slider.valueChanged.connect(self.on_layer_changed)
        self.layer_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                background: #3c3c3c;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6c6c6c;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #8c8c8c;
            }
        """
        )
        self.layer_control_layout.addWidget(self.layer_slider)

        self.current_layer_display = QLabel("0")
        self.current_layer_display.setStyleSheet("color: #c0c0c0; min-width: 20px;")
        self.layer_control_layout.addWidget(self.current_layer_display)

        # Add the layer control layout to the centered container
        controls_layout.addLayout(self.layer_control_layout)
        controls_layout.addStretch()  # Right spacer

        controls_container.setLayout(controls_layout)
        layout.addWidget(controls_container)

        self.att_matrix = None
        self.tokens = None
        self.highlight_patch = None
        self.all_layer_matrices = []  # Store matrices for all layers
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def update_all_layers(self, all_layer_matrices, tokens=None):
        """Update with matrices for all layers for interactive exploration"""
        self.all_layer_matrices = all_layer_matrices
        self.tokens = tokens

        # Update slider range
        if self.all_layer_matrices:
            max_layer = len(self.all_layer_matrices) - 1
            self.layer_slider.setMaximum(max_layer)
            self.layer_slider.setValue(0)  # Start with first layer
            self.current_layer_display.setText("0")

            # Show first layer
            self.update_matrix_display(self.all_layer_matrices[0], tokens)
        else:
            self.layer_slider.setMaximum(0)
            self.update_matrix_display(None, tokens)

    def update_matrix_display(self, att_matrix, tokens=None):
        """Update the matrix display for a specific layer"""
        self.att_matrix = att_matrix

        # Clear the axes completely but preserve layout
        self.ax.clear()
        self.highlight_patch = None

        # Clear any existing colorbar
        if hasattr(self, "cbar") and self.cbar is not None:
            try:
                self.cbar.remove()
            except:
                pass
            self.cbar = None

        # Force consistent figure size and layout with space for colorbar and slider
        self.figure.set_size_inches(7, 4)
        self.figure.subplots_adjust(bottom=0.25, left=0.15, right=0.88, top=0.85)

        if att_matrix is not None and len(att_matrix) > 0:
            im = self.ax.imshow(self.att_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal", origin="lower")
            self.ax.set_xlim(-0.5, att_matrix.shape[1] - 0.5)
            self.ax.set_ylim(-0.5, att_matrix.shape[0] - 0.5)
        else:
            im = None

        # Create a visible colorbar
        if im is not None:
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            cbar = self.figure.colorbar(im, cax=cax, orientation="vertical")
            cbar.set_label("Attention Weight", color="white", fontsize=10, labelpad=10)
            cbar.ax.tick_params(colors="white", labelsize=8)
            cbar.ax.yaxis.set_label_position("right")
            cbar.ax.yaxis.tick_right()

            # Store colorbar for cleanup
            self.cbar = cbar

        # Apply consistent styling
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="white", labelsize=10)
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")

        if tokens and att_matrix is not None and len(tokens) == len(att_matrix):
            self.ax.set_xticks(range(len(tokens)))
            self.ax.set_yticks(range(len(tokens)))
            self.ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
            self.ax.set_yticklabels(tokens, fontsize=9)
            self.ax.set_xlabel("Attended To", color="white", fontsize=11, labelpad=20)
            self.ax.set_ylabel("Attending From", color="white", fontsize=11, labelpad=20)

            self.ax.set_title("Attention Matrix (Hover for weights)", color="white", fontsize=13, pad=20)
        else:
            if att_matrix is None:
                self.ax.text(
                    0.5,
                    0.5,
                    "No attention data\nRun analysis first",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                    color="white",
                    fontsize=14,
                )
            self.ax.set_title("Attention Matrix", color="white", fontsize=13)

        # Force canvas redraw without layout changes
        self.canvas.draw()

    def on_layer_changed(self, value):
        """Handle layer slider changes for real-time exploration"""
        if self.all_layer_matrices and 0 <= value < len(self.all_layer_matrices):
            self.current_layer_display.setText(str(value))
            self.update_matrix_display(self.all_layer_matrices[value], self.tokens)

    def on_hover(self, event):
        if self.att_matrix is not None and event.inaxes == self.ax:
            x = round(event.xdata)
            y = round(event.ydata)
            if 0 <= x < self.att_matrix.shape[1] and 0 <= y < self.att_matrix.shape[0]:
                # Remove previous highlight
                if self.highlight_patch is not None:
                    self.highlight_patch.remove()
                # Add new highlight
                self.highlight_patch = self.ax.add_patch(
                    Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2)
                )
                weight = self.att_matrix[y, x]
                # Get token names for "from" and "to" positions
                from_token = self.tokens[y] if self.tokens and y < len(self.tokens) else f"Token {y}"
                to_token = self.tokens[x] if self.tokens and x < len(self.tokens) else f"Token {x}"
                self.ax.set_title(f"{weight:.1%} from {from_token} to {to_token}")
                self.canvas.draw()
            else:
                # Remove highlight if out of bounds
                if self.highlight_patch is not None:
                    self.highlight_patch.remove()
                    self.highlight_patch = None
                self.ax.set_title("Attention Matrix (Hover for weights)")
                self.canvas.draw()
        else:
            # Remove highlight if not over the axes
            if self.highlight_patch is not None:
                self.highlight_patch.remove()
                self.highlight_patch = None
            if self.att_matrix is not None:
                self.ax.set_title("Attention Matrix (Hover for weights)")
                self.canvas.draw()

    def resizeEvent(self, a0):
        """Handle resize events to adjust figure size dynamically"""
        super().resizeEvent(a0)

        # Adjust figure size based on widget size
        width = self.width()
        height = self.height()

        # Calculate appropriate figure size (in inches) based on widget size
        # Use 100 pixels per inch as a DPI, but constrain to bounds
        fig_width = min(7, max(4, width / 100))  # Max 7 inches wide
        fig_height = min(4, max(3, height / 100))  # Max 4 inches tall

        # Update figure size and layout
        self.figure.set_size_inches(fig_width, fig_height)
        self.figure.subplots_adjust(bottom=0.25, left=0.15, right=0.88, top=0.85)

        # Redraw if the is data
        if self.att_matrix is not None:
            self.canvas.draw()
