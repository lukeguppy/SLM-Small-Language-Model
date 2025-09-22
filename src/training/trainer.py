import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import csv
import math
from typing import Optional, List, Callable, Any

from ..model_config import ModelConfig
from ..core.paths import ModelPaths, DataPaths
from ..core.logger import get_training_logger
from ..services.vocab_service import VocabService
from ..services.model_service import ModelService
from ..services.data_service import DataService


class SentenceDataset(Dataset):
    """Dataset for sentence data with dynamic padding."""

    def __init__(self, sentences: List[List[int]]):
        self.data = sentences

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        tokens = self.data[idx]
        # Inputs: all except last token
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        # Targets: all except first token
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def collate_fn(batch: List[tuple]) -> tuple:
    """Collate function for dynamic padding."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    x_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_pad = torch.zeros(len(ys), max_len, dtype=torch.long)
    mask = torch.zeros(len(xs), max_len, dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        length = len(x)
        x_pad[i, :length] = x
        y_pad[i, :length] = y
        mask[i, :length] = True

    return x_pad, y_pad, mask


class Trainer:
    """
    Refactored trainer using dependency injection.
    No more global variables - all dependencies injected.
    """

    def __init__(
        self, vocab_service: VocabService, model_service: ModelService, data_service: DataService, logger=None
    ):
        if logger is None:
            raise ValueError("logger parameter is required")
        self.vocab_service = vocab_service
        self.model_service = model_service
        self.data_service = data_service
        self.logger = logger

    def train_model(
        self,
        config: ModelConfig,
        callback: Optional[Callable] = None,
        batch_update_callback: Optional[Callable] = None,
        stop_event: Optional[Any] = None,
    ) -> Optional[Any]:
        """Train the model with injected services."""
        try:
            # Phase 1: Setup training environment
            training_context = self._setup_training_environment(config)
            if training_context is None:
                return None

            # Phase 2: Execute training loop
            self._execute_training_loop(config, training_context, callback, batch_update_callback, stop_event)

            # Phase 3: Finalise and cleanup
            return self._finalise_training(config, training_context)

        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            return None

    def _setup_training_environment(self, config: ModelConfig) -> Optional[dict]:
        """Setup the training environment and prepare all necessary components."""
        # Load and prepare vocabulary
        vocab_path = config.vocab_path or DataPaths.get_vocab_path()
        if not self.vocab_service.load_vocabulary(vocab_path):
            self.logger.error(f"Failed to load vocabulary from {vocab_path}")
            return None

        vocab = self.vocab_service.get_vocab()
        vocab_size = self.vocab_service.get_vocab_size()

        # Load and prepare data
        if not self.data_service.load_data(vocab_path=vocab_path):
            self.logger.error("Failed to load training data")
            return None

        tokenised_sentences = self.data_service.filter_sentences_by_vocab(vocab)
        if not tokenised_sentences:
            self.logger.error("No valid training sentences found")
            return None

        # Create data splits and loaders
        data_loaders = self._create_data_loaders(config, tokenised_sentences)
        if data_loaders is None:
            return None

        # Prepare model and optimiser
        model_setup = self._setup_model_and_optimiser(config)
        if model_setup is None:
            return None

        # Setup logging and tracking
        logging_setup = self._setup_training_logging(config)

        return {
            "vocab": vocab,
            "vocab_size": vocab_size,
            "tokenised_sentences": tokenised_sentences,
            **data_loaders,
            **model_setup,
            **logging_setup,
        }

    def _create_data_loaders(self, config: ModelConfig, tokenised_sentences: List[List[int]]) -> Optional[dict]:
        """Create training, validation, and test data loaders."""
        try:
            # Create data splits
            train_sentences, val_sentences, test_sentences = self.data_service.create_train_val_test_split(
                tokenised_sentences, config.train_size, config.val_size, config.test_size, augment=True
            )

            # Create data loaders
            train_loader = DataLoader(
                SentenceDataset(train_sentences), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
            )

            val_loader = DataLoader(
                SentenceDataset(val_sentences), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
            )

            return {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "train_sentences": train_sentences,
                "val_sentences": val_sentences,
            }

        except Exception as e:
            self.logger.error(f"Failed to create data loaders: {e}")
            return None

    def _setup_model_and_optimiser(self, config: ModelConfig) -> Optional[dict]:
        """Setup model, optimiser, and learning rate scheduler."""
        try:
            # Get model from service
            model = self.model_service.get_model()
            if model is None:
                self.logger.error("No model loaded for training")
                return None

            device = next(model.parameters()).device
            optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)

            # Setup learning rate scheduler
            scheduler = self._create_lr_scheduler(config, optimiser)

            return {
                "model": model,
                "device": device,
                "optimiser": optimiser,
                "criterion": criterion,
                "scheduler": scheduler,
            }

        except Exception as e:
            self.logger.error(f"Failed to setup model and optimiser: {e}")
            return None

    def _create_lr_scheduler(self, config: ModelConfig, optimiser):
        """Create learning rate scheduler based on configuration."""
        if config.epochs < 20:
            return torch.optim.lr_scheduler.ConstantLR(optimiser, factor=1.0)
        else:
            # Cosine annealing with warmup
            total_steps = config.epochs * 100  # Approximate based on typical batch count
            warmup_steps = max(int(total_steps * 0.01), 100)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step / max(1, warmup_steps))
                else:
                    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return float(0.5 * (1 + torch.cos(torch.tensor(math.pi * progress))))

            return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    def _setup_training_logging(self, config: ModelConfig) -> dict:
        """Setup training logging and CSV tracking."""
        # Ensure model directory exists
        ModelPaths.ensure_model_dir(config.model_name)
        csv_file = DataPaths.get_training_data_path(config.model_name)

        return {
            "csv_file": csv_file,
            "best_val_loss": float("inf"),
            "csv_initialised": False,
            "step_count": 0,
            "global_batch_count": 0,
            "batch_update_interval": 100,
        }

    def _execute_training_loop(
        self,
        config: ModelConfig,
        context: dict,
        callback: Optional[Callable] = None,
        batch_update_callback: Optional[Callable] = None,
        stop_event: Optional[Any] = None,
    ) -> None:
        """Execute the main training loop."""
        model = context["model"]
        train_loader = context["train_loader"]
        val_loader = context["val_loader"]
        device = context["device"]
        optimiser = context["optimiser"]
        criterion = context["criterion"]
        scheduler = context["scheduler"]
        vocab_size = context["vocab_size"]

        # Log training start
        self._log_training_start(config, context)

        for epoch in range(config.epochs):
            if stop_event and stop_event.is_set():
                self.logger.info("Training stopped by user request")
                break

            # Train for one epoch
            self._train_epoch(
                model,
                train_loader,
                optimiser,
                criterion,
                scheduler,
                config,
                epoch,
                context,
                device,
                vocab_size,
                batch_update_callback,
                stop_event,
            )

            # Validate
            val_metrics = self._validate_epoch(model, val_loader, criterion, device, vocab_size)

            # Log epoch results
            self._log_epoch_results(epoch, config, val_metrics, context)

            # Handle callbacks and model saving
            self._handle_epoch_end(epoch, config, val_metrics, context, callback, model)

    def _train_epoch(
        self,
        model,
        train_loader,
        optimiser,
        criterion,
        scheduler,
        config,
        epoch,
        context,
        device,
        vocab_size,
        batch_update_callback,
        stop_event,
    ):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        batch_count = 0

        for x, y, mask in train_loader:
            if stop_event and stop_event.is_set():
                break

            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimiser.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            if config.epochs >= 20:
                scheduler.step()

            total_loss += loss.item()
            context["step_count"] += 1
            batch_count += 1
            context["global_batch_count"] += 1

            # Batch update callback
            if batch_update_callback and batch_count % context["batch_update_interval"] == 0:
                current_avg_loss = total_loss / batch_count
                total_steps = config.epochs * len(train_loader)
                progress_percent = int((epoch * len(train_loader) + batch_count) / total_steps * 10000)
                batch_update_callback(
                    progress_percent,
                    current_avg_loss,
                    context["global_batch_count"],
                    context["batch_update_interval"],
                    config.epochs * len(train_loader),
                )

        context["avg_train_loss"] = total_loss / len(train_loader)

    def _validate_epoch(self, model, val_loader, criterion, device, vocab_size):
        """Validate the model for one epoch."""
        model.eval()
        val_loss = 0
        top1_correct = 0
        top5_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                logits = model(x, mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()

                # Accuracy calculation
                logits_flat = logits.view(-1, vocab_size)
                y_flat = y.view(-1)
                mask_flat = mask.view(-1)
                valid_mask = mask_flat.bool()

                if valid_mask.any():
                    logits_valid = logits_flat[valid_mask]
                    y_valid = y_flat[valid_mask]

                    # Adaptive top-k based on vocabulary size
                    k = min(5, vocab_size)
                    _, preds = torch.topk(logits_valid, k, dim=-1)
                    correct = preds.eq(y_valid.unsqueeze(1))
                    top1_correct += correct[:, 0].sum().item()
                    if k >= 5:
                        top5_correct += correct.any(dim=1).sum().item()
                    else:
                        top5_correct += correct[:, 0].sum().item()
                    total_tokens += valid_mask.sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        top1_acc = top1_correct / total_tokens if total_tokens > 0 else 0
        top5_acc = top5_correct / total_tokens if total_tokens > 0 else 0

        return {"avg_val_loss": avg_val_loss, "perplexity": perplexity, "top1_acc": top1_acc, "top5_acc": top5_acc}

    def _log_training_start(self, config: ModelConfig, context: dict) -> None:
        """Log training start information."""
        device = context["device"]
        train_sentences = context["train_sentences"]
        val_sentences = context["val_sentences"]

        self.logger.info("Training started...")
        self.logger.info(f"Using device: {device}")
        if torch.cuda.is_available() and device.type == "cuda":
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name(device)}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        self.logger.info(f"Parameters: lr={config.lr}, batch_size={config.batch_size}, epochs={config.epochs}")
        self.logger.info(f"Training data: {len(train_sentences)} sentences")
        self.logger.info(f"Validation data: {len(val_sentences)} sentences")

    def _log_epoch_results(self, epoch: int, config: ModelConfig, val_metrics: dict, context: dict) -> None:
        """Log epoch results and update CSV."""
        avg_train_loss = context["avg_train_loss"]
        avg_val_loss = val_metrics["avg_val_loss"]
        perplexity = val_metrics["perplexity"]
        top1_acc = val_metrics["top1_acc"]
        top5_acc = val_metrics["top5_acc"]

        self.logger.info(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f} | "
            f"Top1 Acc: {top1_acc:.4f} | Top5 Acc: {top5_acc:.4f}"
        )

        # Log to CSV
        csv_file = context["csv_file"]
        if not context["csv_initialised"]:
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "step", "train_loss", "val_loss", "perplexity", "top1_acc", "top5_acc"])
            context["csv_initialised"] = True

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, context["step_count"], avg_train_loss, avg_val_loss, perplexity, top1_acc, top5_acc]
            )

    def _handle_epoch_end(
        self, epoch: int, config: ModelConfig, val_metrics: dict, context: dict, callback: Optional[Callable], model
    ) -> None:
        """Handle end of epoch tasks (callbacks, model saving, text generation)."""
        avg_val_loss = val_metrics["avg_val_loss"]

        # Call epoch callback
        if callback:
            callback(epoch + 1, context["avg_train_loss"], avg_val_loss)

        # Qualitative check: generate text every 5 epochs
        if (epoch + 1) % 5 == 0:
            sample_text = self.generate_text(model, "the")
            self.logger.info(f"Sample generation: {sample_text}")

        # Save best model
        if avg_val_loss < context["best_val_loss"]:
            context["best_val_loss"] = avg_val_loss
            self.model_service.save_model_checkpoint(config.model_name, "best", config)

    def _finalise_training(self, config: ModelConfig, context: dict) -> Optional[Any]:
        """Finalise training and cleanup."""
        model = context["model"]

        # Check if we have epoch results (training wasn't interrupted)
        if "avg_train_loss" in context:
            # Save final model
            self.model_service.save_model_checkpoint(config.model_name, "final", config)

            # Check if final model is the best
            final_val_loss = context.get("final_val_loss", float("inf"))
            final_is_best = abs(final_val_loss - context["best_val_loss"]) < 1e-6

            # Cleanup files based on whether final is best
            self.model_service.cleanup_model_files(config.model_name, final_is_best)

            if final_is_best:
                self.logger.info("Final model is the best - saved as main model, cleaned up best/final files")
            else:
                self.logger.info("Final model is not the best - kept best and final models")
        else:
            # Training stopped mid-epoch, rename best to main if it exists
            best_model_path = ModelPaths.get_best_model_path(config.model_name)
            if os.path.exists(best_model_path):
                self.model_service.save_model_checkpoint(config.model_name, "main", config)
                self.logger.info("Training stopped early - saved best model as main model")

        return model

    def generate_text(self, model: Any, seed_word: str, max_len: int = 10) -> str:
        """Generate text continuation from seed word."""
        model.eval()
        device = next(model.parameters()).device

        # Convert seed word to tokens using vocab service
        tokens = self.vocab_service.text_to_tokens(seed_word)

        for _ in range(max_len):
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            mask = torch.ones(1, len(tokens), dtype=torch.bool).to(device)

            with torch.no_grad():
                logits = model(input_tensor, mask)
                next_token_logits = logits[0, -1]

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            tokens.append(next_token)  # pyright: ignore[reportArgumentType]

            # Stop if we generate padding token (assuming 0 is PAD)
            if next_token == 0:
                break

        # Convert back to text using vocab service
        return self.vocab_service.tokens_to_text(tokens)
