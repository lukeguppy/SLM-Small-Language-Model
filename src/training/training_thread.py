from PyQt5.QtCore import QThread, pyqtSignal
import threading
from .trainer import Trainer
from ..services.vocab_service import VocabService
from ..services.model_service import ModelService
from ..services.data_service import DataService


class TrainingThread(QThread):
    """
    Training thread using dependency injection.
    Creates and manages service instances for training.
    """

    progress = pyqtSignal(int)
    epoch_update = pyqtSignal(int, float, float)
    batch_progress = pyqtSignal(
        int, float, int, int, int
    )  # progress_percent, current_loss, global_batch_count, batch_update_interval, total_batches
    batch_loss_update = pyqtSignal(int, float)  # global_batch_count, current_loss
    finished = pyqtSignal()

    def __init__(self, model, config, logger):
        super().__init__()
        self.model = model
        self.config = config
        self.logger = logger
        self._stop_event = threading.Event()

        # Create service instances with logger injection
        self.vocab_service = VocabService(logger=self.logger)
        self.model_service = ModelService(logger=self.logger)
        self.data_service = DataService(logger=self.logger)

        # Set model in model service
        self.model_service.model = model
        self.model_service.config = config

        # Create trainer with dependency injection
        self.trainer = Trainer(
            vocab_service=self.vocab_service,
            model_service=self.model_service,
            data_service=self.data_service,
            logger=self.logger,
        )

    def requestInterruption(self):
        """Request interruption of the training thread"""
        self._stop_event.set()

    def run(self):
        try:

            def progress_callback(epoch, train_loss, val_loss):
                self.progress.emit(epoch)
                self.epoch_update.emit(epoch, train_loss, val_loss)

            def batch_progress_callback(
                progress_percent, current_loss, global_batch_count=None, batch_update_interval=None, total_batches=None
            ):
                self.batch_progress.emit(
                    progress_percent, current_loss, global_batch_count, batch_update_interval, total_batches
                )
                # Also emit batch loss update for the loss plot
                if global_batch_count is not None:
                    self.batch_loss_update.emit(global_batch_count, current_loss)

            # Use the new trainer with dependency injection
            trained_model = self.trainer.train_model(
                config=self.config,
                callback=progress_callback,
                batch_update_callback=batch_progress_callback,
                stop_event=self._stop_event,
            )

            # Update model service with trained model
            if trained_model is not None:
                self.model_service.model = trained_model

        except Exception as e:
            print(f"Training thread error: {e}")
            import traceback

            traceback.print_exc()
            # Emit finished signal even on error to prevent hanging
            self.finished.emit()
            return

        # Note: Model and meta file saving is now handled by the trainer
        self.finished.emit()
