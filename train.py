"""
train.py

Contains training functions for single training, cross-validation, and continued training.
"""

import os
import time
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import pytorch_lightning as pl
from optuna.exceptions import TrialPruned
from typing import Tuple, Optional
from data import PatchClassificationDataset
from models import build_classifier, LitClassifier
from callbacks import (CleanTQDMProgressBar, TrialFoldProgressCallback,
                       PlotMetricsCallback, OverallProgressCallback,
                       EvaluateTrainMetricsCallback, MasterValidationMetricsCallback,
                       MasterTrainingMetricsCallback)
from utils import load_obj, thai_time

def train_stage(cfg: pl.utilities.parsing.DictConfig, csv_path: str, num_classes: int, stage_name: str,
                trial: Optional[pl.utilities.parsing.DictConfig] = None, suppress_metrics: bool = False,
                trial_number: Optional[int] = None, total_trials: Optional[int] = None,
                fold_number: Optional[int] = None, total_folds: Optional[int] = None) -> Tuple[pl.LightningModule, any]:
    """
    Train a single stage (or fold) using the provided configuration.
    """
    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(full_df, test_size=cfg.data.valid_split, random_state=cfg.training.seed,
                                          stratify=full_df[cfg.data.label_col])
    # Prepare augmentations
    from albumentations import Compose
    train_transforms = Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
    valid_transforms = Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
    train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
    print(f"[INFO] Train dataset size: {len(train_dataset)} | Validation dataset size: {len(valid_dataset)}")
    if trial_number == 1:
        from matplotlib import pyplot as plt
        # Optionally preview augmentation
        indices = np.random.choice(len(train_df), size=min(4, len(train_df)), replace=False)
        for idx in indices:
            row = train_df.iloc[idx]
            img_path = os.path.join(cfg.data.folder_path, row["filename"])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = train_transforms(image=img_rgb)
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.imshow(img_rgb)
            plt.title("Original")
            plt.subplot(1,2,2)
            plt.imshow(augmented["image"].transpose(1,2,0))
            plt.title("Augmented")
            plt.show()
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    # Build model
    model = build_classifier(cfg, num_classes=num_classes)
    if cfg.get("pretrained_ckpt", "None") != "None":
        print(f"Loading pretrained checkpoint from {cfg.pretrained_ckpt}")
        model.load_state_dict(torch.load(cfg.pretrained_ckpt))
    lit_model = LitClassifier(cfg=cfg, model=model, num_classes=num_classes)
    # Generate a unique stage ID using Thai time and milliseconds
    stage_id = f"{stage_name}_{thai_time().strftime('%Y%m%d-%H%M%S')}_{int(time.time()*1000)}"
    global BASE_SAVE_DIR
    save_dir = os.path.join(BASE_SAVE_DIR, stage_id)
    os.makedirs(save_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}")
    max_epochs = cfg.training.tuning_epochs_detection
    # Prepare callbacks
    callbacks = []
    if not suppress_metrics:
        callbacks.append(PlotMetricsCallback())
        callbacks.append(EvaluateTrainMetricsCallback(train_loader))
    callbacks.append(MasterTrainingMetricsCallback(base_dir=BASE_SAVE_DIR, trial_number=trial_number, fold_number=fold_number))
    callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"))
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=f"{stage_name}-" + "{epoch:02d}-{val_acc:.4f}"
    ))
    callbacks.append(OverallProgressCallback())
    callbacks.append(TrialFoldProgressCallback(trial_number=trial_number, total_trials=total_trials,
                                                fold_number=fold_number, total_folds=total_folds))
    if trial is not None:
        class OptunaReportingCallback(pl.Callback):
            def __init__(self, trial, metric_name="val_acc"):
                super().__init__()
                self.trial = trial
                self.metric_name = metric_name
            def on_validation_epoch_end(self, trainer, pl_module):
                val_metric = trainer.callback_metrics.get(self.metric_name)
                if val_metric is not None:
                    self.trial.report(val_metric.item(), step=trainer.current_epoch)
                    if self.trial.should_prune():
                        raise TrialPruned()
        callbacks.append(OptunaReportingCallback(trial, metric_name="val_acc"))
    callbacks.append(MasterValidationMetricsCallback(base_dir=BASE_SAVE_DIR, trial_number=trial_number, fold_number=fold_number))
    callbacks.append(CleanTQDMProgressBar())
    trainer = pl.Trainer(max_epochs=max_epochs,
                         devices=cfg.trainer.devices,
                         accelerator=cfg.trainer.accelerator,
                         precision=cfg.trainer.precision,
                         gradient_clip_val=cfg.trainer.get("gradient_clip_val", None),
                         logger=logger, callbacks=callbacks, enable_model_summary=False)
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return lit_model, trainer.callback_metrics.get("val_acc")


def train_with_cross_validation(cfg: DictConfig, csv_path: str, num_classes: int, stage_name: str) -> Tuple[pl.LightningModule, float]:
    full_df = pd.read_csv(csv_path)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cfg.training.num_folds, shuffle=True, random_state=cfg.training.seed)
    val_scores = []
    fold_models = []
    for fold, (train_idx, valid_idx) in enumerate(skf.split(full_df, full_df[cfg.data.label_col])):
        fold_num = fold + 1
        print(f"Fold {fold_num}/{cfg.training.num_folds}")
        train_df = full_df.iloc[train_idx]
        valid_df = full_df.iloc[valid_idx]
        lit_model, val_acc = train_stage(cfg, csv_path, num_classes, stage_name=f"{stage_name}_fold{fold_num}",
                                         fold_number=fold_num, total_folds=cfg.training.num_folds)
        score = val_acc.item() if val_acc is not None else 0.0
        print(f"Fold {fold_num} validation accuracy: {score:.4f}")
        val_scores.append(score)
        fold_models.append(lit_model)
    avg_score = float(np.mean(val_scores))
    print(f"Average cross-validation accuracy: {avg_score:.4f}")
    best_idx = int(np.argmax(val_scores))
    return fold_models[best_idx], avg_score


def repeated_cross_validation(cfg: DictConfig, csv_path: str, num_classes: int, stage_name: str, repeats: int) -> Tuple[pl.LightningModule, float]:
    all_scores = []
    best_models = []
    for r in range(repeats):
        print(f"\n=== Repeated CV run {r+1}/{repeats} ===")
        model_cv, avg_score = train_with_cross_validation(cfg, csv_path, num_classes, stage_name)
        all_scores.append(avg_score)
        best_models.append(model_cv)
    overall_avg = float(np.mean(all_scores))
    overall_std = float(np.std(all_scores))
    print(f"\nRepeated CV over {repeats} runs: {overall_avg:.4f} ± {overall_std:.4f}")
    best_idx = int(np.argmax(all_scores))
    return best_models[best_idx], overall_avg


def continue_training(lit_model: pl.LightningModule, cfg: DictConfig, csv_path: str, num_classes: int, stage_name: str) -> pl.LightningModule:
    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(full_df, test_size=cfg.data.valid_split, random_state=cfg.training.seed,
                                          stratify=full_df[cfg.data.label_col])
    from albumentations import Compose
    train_transforms = Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs])
    valid_transforms = Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
    train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    additional_epochs = cfg.training.additional_epochs_detection
    stage_id = f"{stage_name}_continued_{thai_time().strftime('%Y%m%d-%H%M%S')}_{int(time.time()*1000)}"
    global BASE_SAVE_DIR
    save_dir = os.path.join(BASE_SAVE_DIR, stage_id)
    os.makedirs(save_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}_continued")
    master_callback = MasterValidationMetricsCallback(base_dir=BASE_SAVE_DIR)
    callbacks = [
        PlotMetricsCallback(),
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        pl.callbacks.ModelCheckpoint(dirpath=save_dir, monitor="val_acc", mode="max",
                        filename=f"{stage_name}_continued-" + "{epoch:02d}-{val_acc:.4f}"),
        OverallProgressCallback(),
        master_callback,
        CleanTQDMProgressBar()
    ]
    trainer = pl.Trainer(max_epochs=additional_epochs, devices=cfg.trainer.devices, accelerator=cfg.trainer.accelerator,
                         precision=cfg.trainer.precision, logger=logger, callbacks=callbacks, enable_model_summary=False)
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return lit_model
