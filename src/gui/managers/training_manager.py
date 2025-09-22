import time
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from src.model_config import ModelConfig
from src.training.training_thread import TrainingThread


class TrainingManager(QObject):
    """Manager for training orchestration and state management"""

    # Signals
    epoch_completed = pyqtSignal(int, float, float)  # epoch, train_loss, val_loss
    batch_completed = pyqtSignal(
        int, float, int, bool, int, float, int
    )  # progress%, loss, batch_idx, is_interpolated, total_batches, interval_time, batch_update_interval
    batch_loss_updated = pyqtSignal(int, float)  # batch_count, loss
    progress_updated = pyqtSignal(int)  # epoch
    training_finished = pyqtSignal()
    time_remaining_updated = pyqtSignal(str)  # time remaining as string

    def __init__(self, model_manager, logger):
        super().__init__()
        self.model_manager = model_manager
        self.logger = logger
        self.training_thread = None
        self.is_training = False

        # Time tracking variables (from old system)
        self.training_start_time = None
        self.batch_times = []  # Time for each completed batch
        self.last_batch_start = None  # Start time of current batch
        self.batches_completed = 0  # Running count of completed batches
        self.total_batches_expected = 0  # Total batches expected for training
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_countdown)
        self.base_status = ""
        self.countdown_finished = False

        # Countdown timer variables
        self.last_estimate_batch = 0  # Last batch where estimate was updated
        self.current_countdown_seconds = 0  # Current countdown value in seconds
        self.estimate_update_interval = 100  # Update estimate every 100 batches

    def start_training(self, config: ModelConfig):
        """Start training with given ModelConfig"""
        if self.is_training:
            return False

        model = self.model_manager.get_model_service().get_model()
        if model is None:
            raise ValueError("No model loaded for training")

        # Create training thread with config and logger
        self.training_thread = TrainingThread(model, config, self.logger)

        # Connect signals
        self.training_thread.finished.connect(self._on_training_finished)
        self.training_thread.epoch_update.connect(self._on_epoch_update)
        self.training_thread.batch_progress.connect(self._on_batch_progress)
        self.training_thread.batch_loss_update.connect(self._on_batch_loss_update)
        self.training_thread.progress.connect(self._on_progress)

        # Initialise time tracking
        self.training_start_time = time.time()
        self.total_batches_expected = (
            config.epochs * config.train_size
        ) // config.batch_size

        # Reset training state variables
        self.batch_times = []
        self.last_batch_start = time.time()  # Start timing first batch
        self.batches_completed = 0
        self.last_estimate_batch = 0
        self.current_countdown_seconds = 0
        self.countdown_finished = False

        # Start the countdown timer
        self.timer.start(1000)  # updates each second

        # Start training
        self.training_thread.start()
        self.is_training = True
        return True

    def stop_training(self):
        """Stop current training"""
        if self.training_thread:
            if self.training_thread.isRunning():
                # Request termination
                self.training_thread.requestInterruption()
                # Give it a moment to finish gracefully
                if not self.training_thread.wait(5000):  # Wait up to 5 seconds
                    print(
                        "Training thread didn't respond to interruption, terminating forcefully"
                    )
                    self.training_thread.terminate()
                    # Wait another 2 seconds for termination
                    self.training_thread.wait(2000)
            # Clean up thread reference
            self.training_thread = None

        # Stop timers
        self.timer.stop()

        # Reset state
        self.is_training = False
        self.training_start_time = None
        self.batch_times = []
        self.last_batch_start = None
        self.batches_completed = 0
        self.current_countdown_seconds = 0
        self.countdown_finished = False

    def finish_training(self):
        """Handle training completion and show total time taken"""
        # Stop timers
        self.timer.stop()

        # Calculate total training time
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            time_str = self.format_time(int(total_time))
            self.time_remaining_updated.emit(f"Completed in {time_str}")
        else:
            self.time_remaining_updated.emit("Completed")

        # Reset state
        self.is_training = False
        self.training_start_time = None
        self.batch_times = []
        self.last_batch_start = None
        self.batches_completed = 0
        self.current_countdown_seconds = 0
        self.countdown_finished = True

    def get_training_status(self):
        """Get current training status"""
        return {"is_training": self.is_training, "thread": self.training_thread}

    def _on_epoch_update(self, epoch, train_loss, val_loss):
        """Handle epoch completion"""
        self.epoch_completed.emit(epoch, train_loss, val_loss)

    def _on_batch_progress(
        self,
        progress_percent,
        current_loss,
        global_batch_index,
        batch_update_interval,
        total_batches=None,
    ):
        """Handle batch progress with timing tracking"""
        # Update total_batches_expected if provided
        if total_batches is not None:
            self.total_batches_expected = total_batches

        # Track batch completion times for accurate time estimation
        current_time = time.time()

        # Calculate time for this update interval
        interval_time = 0
        if self.last_batch_start:
            # Calculate time for this update interval
            interval_time = current_time - self.last_batch_start

            # If there is a batch_update_interval, calculate per-batch time
            if batch_update_interval and batch_update_interval > 1:
                batch_time = interval_time / batch_update_interval
                # Record individual batch times for the interval
                for _ in range(batch_update_interval):
                    self.batch_times.append(batch_time)
            else:
                # Fallback: assume interval is 1 batch
                batch_time = interval_time
                self.batch_times.append(batch_time)

            # Use global batch index if provided, otherwise increment
            if global_batch_index is not None:
                self.batches_completed = global_batch_index
            else:
                self.batches_completed += 1

        # Start timing next batch
        self.last_batch_start = current_time

        # Emit batch progress for status update
        self.batch_completed.emit(
            progress_percent,
            current_loss,
            global_batch_index,
            False,
            self.total_batches_expected,
            interval_time,
            batch_update_interval,
        )

        # Update countdown only when updating estimate (every 100 batches)
        if (
            self.batches_completed - self.last_estimate_batch
        ) >= self.estimate_update_interval:
            self.update_countdown()

    def _on_batch_loss_update(self, batch_count, loss):
        """Handle batch loss update for plotting"""
        self.batch_loss_updated.emit(batch_count, loss)

    def _on_progress(self, epoch):
        """Handle progress update"""
        self.progress_updated.emit(epoch)

    def update_countdown(self):
        """Update time remaining based on batch-level performance data"""
        if not self.training_start_time:
            self.time_remaining_updated.emit("Not training")
            return

        current_time = time.time()
        elapsed = current_time - self.training_start_time
        elapsed_str = self.format_time(elapsed)

        # If no batches completed yet, show elapsed time and estimating
        if len(self.batch_times) == 0:
            self.time_remaining_updated.emit(
                f"Elapsed: {elapsed_str}\nEstimated time remaining: estimating..."
            )
            return

        # Check if we should update the estimate (every 100 batches)
        should_update_estimate = (
            self.batches_completed - self.last_estimate_batch
        ) >= self.estimate_update_interval

        if should_update_estimate and len(self.batch_times) > 0:
            # Update estimate using moving average of recent batch times (last 20 batches)
            recent_batches = self.batch_times[-20:]  
            avg_batch_time = sum(recent_batches) / len(recent_batches)

            # Calculate remaining time: avg_batch_time * remaining_batches
            batches_remaining = max(
                0, self.total_batches_expected - self.batches_completed
            )
            est_remaining = avg_batch_time * batches_remaining

            if est_remaining > 0:
                # Set the countdown value
                self.current_countdown_seconds = int(est_remaining)
                self.last_estimate_batch = self.batches_completed
                remaining_str = self.format_time(self.current_countdown_seconds)
                self.time_remaining_updated.emit(
                    f"Elapsed: {elapsed_str}\nEstimated time remaining: {remaining_str}"
                )
                return

        # Decrement countdown value every second
        if self.current_countdown_seconds > 0:
            self.current_countdown_seconds -= 1
            remaining_str = self.format_time(self.current_countdown_seconds)
            self.time_remaining_updated.emit(
                f"Elapsed: {elapsed_str}\nEstimated time remaining: {remaining_str}"
            )
            return

        # Fallback if no batch data yet or countdown finished
        self.time_remaining_updated.emit(
            f"Elapsed: {elapsed_str}\nEstimated time remaining: estimating..."
        )

    def format_time(self, seconds):
        """Format seconds into human readable time string"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds_remain = int(seconds % 60)
            return f"{minutes}m {seconds_remain}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _on_training_finished(self):
        """Handle training completion"""
        self.is_training = False
        self.training_thread = None
        self.training_finished.emit()
