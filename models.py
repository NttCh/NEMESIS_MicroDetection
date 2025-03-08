"""
models.py

Contains functions for building models and the PyTorch Lightning module.
"""

import torch.nn as nn
import torchvision
from typing import Any
from utils import load_obj
import pytorch_lightning as pl


def build_classifier(cfg: Any, num_classes: int) -> nn.Module:
    """
    Build a classifier model using a backbone defined in the configuration.
    """
    backbone_cls = load_obj(cfg.model.backbone.class_name)
    model = backbone_cls(**cfg.model.backbone.params)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class LitClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for classification.
    """
    def __init__(self, cfg: Any, model: nn.Module, num_classes: int) -> None:
        super(LitClassifier, self).__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Any:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Any:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer_cls = load_obj(self.cfg.optimizer.class_name)
        optimizer_params = self.cfg.optimizer.params.copy()
        # Remove keys not supported by optimizer
        if "gradient_clip_val" in optimizer_params:
            del optimizer_params["gradient_clip_val"]
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        scheduler_cls = load_obj(self.cfg.scheduler.class_name)
        scheduler_params = self.cfg.scheduler.params
        scheduler = scheduler_cls(optimizer, **scheduler_params)
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": self.cfg.scheduler.step,
            "monitor": self.cfg.scheduler.monitor
        }]
