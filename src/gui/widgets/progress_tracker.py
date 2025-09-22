import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from ..styles import PROGRESS_STYLE


class ProgressTracker(QWidget):
    """Progress bar and time estimation display"""

    def __init__(self):
        super().__init__()
        self.total_batches = 0
        self.current_batch_estimate = 0
        self.batch_times = []
        self.last_update_time = 0
        self.last_timer_update_time = 0
        self.progress_interpolation_interval = 100

        # Create progress bar
        self.progress_bar = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the progress tracking UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

    def set_progress_bar(self, progress_bar):
        """Set the progress bar widget to control"""
        self.progress_bar = progress_bar
        if self.progress_bar:
            self.progress_bar.setStyleSheet(PROGRESS_STYLE)

    def start_tracking(self, total_batches):
        """Start tracking progress for training"""
        self.total_batches = total_batches
        self.current_batch_estimate = 0
        self.batch_times = []
        self.last_update_time = 0
        self.last_timer_update_time = 0

        if self.progress_bar:
            self.progress_bar.setMaximum(10000)  # 0.01% precision
            self.progress_bar.setValue(0)

    def update_batch_progress(self, batch_idx, interval_time, batch_update_interval=None):
        """Update progress based on batch completion"""
        if not self.progress_bar:
            return

        # Record batch completion time
        current_time = time.time()

        # Calculate per-batch time if interval provided
        if batch_update_interval and batch_update_interval > 1:
            batch_time = interval_time / batch_update_interval
            for _ in range(batch_update_interval):
                self.batch_times.append(batch_time)
        else:
            batch_time = interval_time
            self.batch_times.append(batch_time)

        # Update batch estimate
        self.current_batch_estimate = batch_idx

        # Calculate progress percentage
        if self.total_batches > 0:
            progress_percentage = int((batch_idx / self.total_batches) * 10000)
            self.progress_bar.setValue(progress_percentage)
            self.progress_bar.repaint()

        self.last_update_time = current_time

    def update_progress_smoothly(self):
        """Update progress bar with smooth interpolation"""
        if not self.progress_bar or self.total_batches == 0 or len(self.batch_times) == 0:
            return

        current_time = time.time()

        # Use average batch time for interpolation
        recent_batches = self.batch_times[-20:]  # Last 20 batches
        avg_batch_time = sum(recent_batches) / len(recent_batches)

        time_since_last_update = current_time - self.last_timer_update_time if self.last_timer_update_time > 0 else 0

        # Estimate batches completed in this interval
        estimated_batches_this_interval = time_since_last_update / avg_batch_time

        # Update current batch estimate
        self.current_batch_estimate = min(
            self.current_batch_estimate + self.progress_interpolation_interval,
            self.current_batch_estimate + estimated_batches_this_interval,
        )

        # Calculate interpolated progress
        progress_percentage = int((self.current_batch_estimate / self.total_batches) * 10000)
        self.progress_bar.setValue(progress_percentage)
        self.progress_bar.repaint()

        self.last_timer_update_time = current_time

    def stop_tracking(self):
        """Stop progress tracking"""
        self.total_batches = 0
        self.current_batch_estimate = 0
        self.batch_times = []
        self.last_update_time = 0
        self.last_timer_update_time = 0

        if self.progress_bar:
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(100)  # Reset to default maximum

    def get_progress_percentage(self):
        """Get current progress as percentage"""
        if self.total_batches == 0:
            return 0.0
        return (self.current_batch_estimate / self.total_batches) * 100.0

    def get_estimated_time_remaining(self):
        """Get estimated time remaining in seconds"""
        if len(self.batch_times) == 0 or self.total_batches == 0:
            return None

        remaining_batches = self.total_batches - self.current_batch_estimate
        if remaining_batches <= 0:
            return 0

        avg_batch_time = sum(self.batch_times[-20:]) / len(self.batch_times[-20:])
        return avg_batch_time * remaining_batches
