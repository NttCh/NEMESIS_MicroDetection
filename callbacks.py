"""
callbacks.py

Defines custom PyTorch Lightning callbacks.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from typing import List

class CleanTQDMProgressBar(TQDMProgressBar):
    """Custom progress bar that removes itself after training."""
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar

class TrialFoldProgressCallback(pl.Callback):
    """Logs which trial/fold is running."""
    def __init__(self, trial_number=None, total_trials=None, fold_number=None, total_folds=None):
        super().__init__()
        self.trial_number = trial_number
        self.total_trials = total_trials
        self.fold_number = fold_number
        self.total_folds = total_folds

    def on_train_start(self, trainer, pl_module):
        msgs = []
        if self.trial_number is not None and self.total_trials is not None:
            msgs.append(f"Trial {self.trial_number}/{self.total_trials}")
        if self.fold_number is not None and self.total_folds is not None:
            msgs.append(f"Fold {self.fold_number}/{self.total_folds}")
        if msgs:
            print(" | ".join(msgs))

class PlotMetricsCallback(pl.Callback):
    """Plots training and validation metrics at the end of training."""
    def __init__(self) -> None:
        super().__init__()
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1  # 1-based numbering
        self.epochs.append(epoch)
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        val_acc = trainer.callback_metrics.get("val_acc")
        self.train_losses.append(train_loss.item() if train_loss is not None else float('nan'))
        self.val_losses.append(val_loss.item() if val_loss is not None else float('nan'))
        self.train_accs.append(train_acc.item() if train_acc is not None else float('nan'))
        self.val_accs.append(val_acc.item() if val_acc is not None else float('nan'))

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(self.epochs, self.train_losses, label="Train Loss", marker="o")
        axs[0].plot(self.epochs, self.val_losses, label="Validation Loss", marker="o")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Loss vs. Epoch")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(self.epochs, self.train_accs, label="Train Accuracy", marker="o")
        axs[1].plot(self.epochs, self.val_accs, label="Validation Accuracy", marker="o")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_title("Accuracy vs. Epoch")
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
        save_path = os.path.join(trainer.logger.log_dir, "metrics_plot.png")
        plt.savefig(save_path)
        print(f"[PlotMetricsCallback] Saved metrics plot to {save_path}")
        plt.show()

class OverallProgressCallback(pl.Callback):
    """Prints progress information at the start of each epoch."""
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.total_epochs = trainer.max_epochs

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        remaining = self.total_epochs - trainer.current_epoch
        print(f"[OverallProgressCallback] Epoch {epoch}/{self.total_epochs} - Remaining epochs: {remaining}")

class EvaluateTrainMetricsCallback(pl.Callback):
    """Evaluates training set performance at the end of each epoch."""
    def __init__(self, train_dataloader) -> None:
        super().__init__()
        self.train_dataloader = train_dataloader

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.train_dataloader:
                images = images.to(pl_module.device)
                labels = labels.to(pl_module.device)
                logits = pl_module(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        train_acc_eval = correct / total if total > 0 else 0.0
        trainer.logger.log_metrics({"train_acc_eval": train_acc_eval}, step=trainer.current_epoch)
        pl_module.train()

class MasterValidationMetricsCallback(pl.Callback):
    """Records validation metrics to an Excel file."""
    def __init__(self, base_dir: str, trial_number=None, fold_number=None):
        super().__init__()
        self.excel_path = os.path.join(base_dir, "all_eval_metrics.xlsx")
        self.trial_number = trial_number
        self.fold_number = fold_number
        self.rows = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.eval()
        val_loader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        all_preds, all_labels, all_loss, count = [], [], 0.0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(pl_module.device)
                labels = labels.to(pl_module.device)
                logits = pl_module(images)
                loss = criterion(logits, labels)
                all_loss += loss.item()
                count += 1
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = all_loss / count if count > 0 else 0.0
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch = trainer.current_epoch + 1
        row = {
            'trial': self.trial_number,
            'fold': self.fold_number,
            'epoch': epoch,
            'val_loss': avg_loss,
            'val_acc': acc,
            'val_prec': prec,
            'val_recall': rec,
            'val_f1': f1
        }
        self.rows.append(row)
        pl_module.train()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if os.path.exists(self.excel_path):
            old_df = pd.read_excel(self.excel_path)
            new_df = pd.DataFrame(self.rows)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = pd.DataFrame(self.rows)
        combined.to_excel(self.excel_path, index=False)
        print(f"[MasterValidationMetricsCallback] Logged validation metrics to {self.excel_path}")

class MasterTrainingMetricsCallback(pl.Callback):
    """Records training metrics to an Excel file."""
    def __init__(self, base_dir: str, trial_number=None, fold_number=None):
        super().__init__()
        self.excel_path = os.path.join(base_dir, "all_train_metrics.xlsx")
        self.trial_number = trial_number
        self.fold_number = fold_number
        self.rows = []

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        train_loader = trainer.train_dataloaders[0] if isinstance(trainer.train_dataloaders, list) else trainer.train_dataloaders
        all_preds, all_labels, all_loss, count = [], [], 0.0, 0
        criterion = pl_module.criterion
        pl_module.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(pl_module.device)
                labels = labels.to(pl_module.device)
                logits = pl_module(images)
                loss = criterion(logits, labels)
                all_loss += loss.item()
                count += 1
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_loss = all_loss / count if count > 0 else 0.0
        acc = accuracy_score(all_labels, all_preds)
        epoch = trainer.current_epoch + 1
        row = {
            'trial': self.trial_number,
            'fold': self.fold_number,
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_acc': acc
        }
        self.rows.append(row)
        pl_module.train()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if os.path.exists(self.excel_path):
            old_df = pd.read_excel(self.excel_path)
            new_df = pd.DataFrame(self.rows)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = pd.DataFrame(self.rows)
        combined.to_excel(self.excel_path, index=False)
        print(f"[MasterTrainingMetricsCallback] Logged training metrics to {self.excel_path}")
