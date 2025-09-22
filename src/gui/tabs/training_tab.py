from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
)
from PyQt5.QtCore import QTimer
import torch
from ..main_gui import CustomProgressBar
from ..widgets.training_controls import TrainingControls
from ..widgets.progress_tracker import ProgressTracker
from ..widgets.loss_plot_widget import LossPlotWidget
from ..styles import (
    GROUP_STYLE,
    BUTTON_STYLE,
    STOP_BUTTON_STYLE,
    TRAINING_STATUS_STYLE,
    TIME_REMAINING_STYLE,
    MAIN_WINDOW_STYLE,
)


def get_cuda_info():
    """Get detailed CUDA information for diagnostics"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", "Not available"),
        "cudnn_version": getattr(torch.backends.cudnn, "version", "Not available"),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
    }

    if torch.cuda.is_available():
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_memory"] = (
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        except Exception as e:
            info["device_error"] = str(e)

    return info


class TrainingTab(QWidget):
    def __init__(self, training_manager, model_manager, config):
        super().__init__()
        self.training_manager = training_manager
        self.model_manager = model_manager
        self.config = config
        self.setStyleSheet(MAIN_WINDOW_STYLE)
        self.progress_tracker = ProgressTracker()
        self.progress_update_timer = QTimer()
        self.progress_update_timer.timeout.connect(
            self.progress_tracker.update_progress_smoothly
        )
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Training parameters
        self.training_controls = TrainingControls(self.config)
        layout.addWidget(self.training_controls)

        # Training controls
        controls_group = QGroupBox("Training Controls")
        controls_group.setStyleSheet(GROUP_STYLE)
        controls_layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setStyleSheet(BUTTON_STYLE)
        button_layout.addWidget(self.train_button)

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setStyleSheet(STOP_BUTTON_STYLE)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        # Add CUDA diagnostic button
        self.cuda_button = QPushButton("CUDA Info")
        self.cuda_button.clicked.connect(self.show_cuda_info)
        self.cuda_button.setStyleSheet(BUTTON_STYLE)
        button_layout.addWidget(self.cuda_button)

        controls_layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = CustomProgressBar()
        self.progress_tracker.set_progress_bar(self.progress_bar)
        controls_layout.addWidget(self.progress_bar)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Training status and time remaining
        status_container = QWidget()
        status_layout = QHBoxLayout()
        status_container.setLayout(status_layout)

        # Training status
        status_group = QGroupBox("Training Status")
        status_group.setStyleSheet(GROUP_STYLE)
        status_inner_layout = QVBoxLayout()

        # Enhanced CUDA status display
        cuda_info = get_cuda_info()
        if cuda_info["cuda_available"]:
            device_info = cuda_info.get("device_name", "Unknown GPU")
            memory_info = cuda_info.get("device_memory", "")
            cuda_status = f"CUDA available: {device_info}"
            if memory_info:
                cuda_status += f" ({memory_info})"
        else:
            cuda_status = "CUDA not available, using CPU"

        pytorch_info = f"PyTorch {cuda_info['pytorch_version']}"
        if cuda_info["cuda_version"] != "Not available":
            pytorch_info += f", CUDA {cuda_info['cuda_version']}"

        self.training_status = QLabel(f"Ready to train\n{cuda_status}\n{pytorch_info}")
        self.training_status.setStyleSheet(TRAINING_STATUS_STYLE)
        status_inner_layout.addWidget(self.training_status)

        status_group.setLayout(status_inner_layout)
        status_layout.addWidget(status_group, 3)

        # Time remaining
        time_group = QGroupBox("Time Remaining")
        time_group.setStyleSheet(GROUP_STYLE)
        time_inner_layout = QVBoxLayout()

        self.time_remaining_label = QLabel("Not training")
        self.time_remaining_label.setStyleSheet(TIME_REMAINING_STYLE)
        time_inner_layout.addWidget(self.time_remaining_label)

        time_group.setLayout(time_inner_layout)
        status_layout.addWidget(time_group, 1)

        layout.addWidget(status_container)

        # Loss plot
        plot_group = QGroupBox("Training Loss")
        plot_group.setStyleSheet(GROUP_STYLE)
        plot_layout = QVBoxLayout()

        self.loss_plot_widget = LossPlotWidget()
        plot_layout.addWidget(self.loss_plot_widget)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        self.setLayout(layout)

    def start_training(self):
        # Get parameters from controls
        config = self.training_controls.get_training_parameters()
        # Update vocab_size in config
        vocab_size = self.model_manager.vocab_manager.get_vocab_size()
        config.vocab_size = vocab_size

        # Create new model for training
        from ...core.model import SmallTransformer

        self.model_manager.get_model_service().model = SmallTransformer(config)

        # Enhanced CUDA detection and device selection
        cuda_info = get_cuda_info()
        device = torch.device("cuda" if cuda_info["cuda_available"] else "cpu")

        if cuda_info["cuda_available"]:
            device_info = cuda_info.get("device_name", "Unknown GPU")
            memory_info = cuda_info.get("device_memory", "")
            device_status = f"CUDA GPU: {device_info}"
            if memory_info:
                device_status += f" ({memory_info})"
        else:
            device_status = "CPU (CUDA not available)"

        # Move model to appropriate device
        self.model_manager.get_model_service().model.to(device)

        # Start training via manager
        self.training_manager.start_training(config)
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # Disable parameter controls during training
        self.training_controls.set_parameters_enabled(False)
        self.progress_tracker.stop_tracking()  # Reset
        self.progress_update_timer.start(100)  # Start smooth updates

        self.training_status.setText(
            f"Training started on {device_status}\n"
            f"Device: {device}\n"
            f"Parameters: lr={config.lr}, batch_size={config.batch_size}, epochs={config.epochs}\n"
            f"Data sizes: train={config.train_size}, val={config.val_size}, test={config.test_size}"
        )
        self.time_remaining_label.setText("Calculating...")

    def stop_training(self):
        self.training_manager.stop_training()
        self.progress_update_timer.stop()
        self.progress_tracker.stop_tracking()
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.training_status.setText("Training stopped by user.")
        self.time_remaining_label.setText("Not training")
        # Re-enable parameter controls
        self.training_controls.set_parameters_enabled(True)

    def update_training_status(self, epoch, train_loss, val_loss):
        train_perplexity = 2**train_loss
        val_perplexity = 2**val_loss
        self.training_status.setText(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f} (PPL: {train_perplexity:.2f}) | "
            f"Val Loss: {val_loss:.4f} (PPL: {val_perplexity:.2f})"
        )
        # Update loss plot with epoch data
        self.loss_plot_widget.update_plot_epoch(epoch, train_loss, val_loss)

    def update_training_status_epoch(self, epoch):
        self.training_status.setText(f"Epoch {epoch}")

    def update_batch_progress(
        self,
        progress_percent,
        current_loss,
        global_batch_index,
        is_interpolated=False,
        total_batches=None,
        interval_time=None,
        batch_update_interval=None,
    ):
        if not is_interpolated:
            # Real update: use ProgressTracker
            if total_batches and interval_time is not None and batch_update_interval:
                if self.progress_tracker.total_batches == 0:
                    self.progress_tracker.start_tracking(total_batches)
                self.progress_tracker.update_batch_progress(
                    global_batch_index, interval_time, batch_update_interval
                )
        # Interpolated updates are handled by ProgressTracker internally

    def update_time_remaining(self, time_str):
        """Update the time remaining display"""
        self.time_remaining_label.setText(time_str)

    def show_cuda_info(self):
        """Show detailed CUDA information"""
        cuda_info = get_cuda_info()

        info_lines = [
            "CUDA/PyTorch Information:",
            f"PyTorch Version: {cuda_info['pytorch_version']}",
            f"CUDA Available: {cuda_info['cuda_available']}",
        ]

        if cuda_info["cuda_available"]:
            info_lines.extend(
                [
                    f"CUDA Version: {cuda_info['cuda_version']}",
                    f"cuDNN Version: {cuda_info['cudnn_version']}",
                    f"GPU Device Count: {cuda_info['device_count']}",
                    f"Current Device: {cuda_info['current_device']}",
                ]
            )

            if "device_name" in cuda_info:
                info_lines.append(f"GPU Name: {cuda_info['device_name']}")
            if "device_memory" in cuda_info:
                info_lines.append(f"GPU Memory: {cuda_info['device_memory']}")

            if "device_error" in cuda_info:
                info_lines.append(f"GPU Error: {cuda_info['device_error']}")
        else:
            info_lines.extend(
                [
                    "CUDA Status: Not available",
                    "Reason: PyTorch installed without CUDA support or CUDA drivers not found",
                    "Training will use CPU (slower but functional)",
                ]
            )

        # Show in training status
        self.training_status.setText("\n".join(info_lines))

    def on_training_finished(self):
        """Handle training completion"""
        self.progress_update_timer.stop()
        self.progress_tracker.stop_tracking()
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.training_status.setText("Training completed successfully!")

        # Use training manager's finish_training method to show completion time
        self.training_manager.finish_training()

        # Re-enable parameter controls
        self.training_controls.set_parameters_enabled(True)
