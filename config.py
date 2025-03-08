"""
config.py

This module contains the global configuration using OmegaConf.
"""

from omegaconf import OmegaConf, DictConfig

cfg_dict = {
    "stage_detection": True,
    "general": {
        "save_dir": "baclogs2",         # Base directory for logs/checkpoints
        "project_name": "bacteria"
    },
    "trainer": {
        "devices": 1,
        "accelerator": "auto",
        "precision": "16-mixed",
        "gradient_clip_val": 0.5  # For Trainer (this key will be removed from optimizer params later)
    },
    "training": {
        "seed": 666,
        "mode": "max",
        "tuning_epochs_detection": 5,    # For hyperparameter tuning (quick test)
        "additional_epochs_detection": 5,  # For continued training
        "cross_validation": True,         # Enable cross-validation
        "num_folds": 2,                   # Number of folds for CV
        "repeated_cv": 2                  # Repeat CV 2 times
    },
    "optimizer": {
        "class_name": "torch.optim.AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.001,
            "gradient_clip_val": 0.0   # This key will be removed before creating the optimizer
        }
    },
    "scheduler": {
        "class_name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "step": "epoch",
        "monitor": "val_acc",
        "params": {
            "mode": "max",
            "factor": 0.1,
            "patience": 10
        }
    },
    "model": {
        "backbone": {
            "class_name": "torchvision.models.resnet50",
            "params": {
                "weights": "IMAGENET1K_V1"
            }
        }
    },
    "data": {
        "detection_csv": "/kaggle/working/train.csv",  # Adjust path as needed
        "folder_path": "/kaggle/input/bacdataset/Data",  # Adjust path as needed
        "num_workers": 3,
        "batch_size": 4,
        "label_col": "label",
        "valid_split": 0.2
    },
    "augmentation": {
        "train": {
            "augs": [
                {"class_name": "albumentations.Resize", "params": {"height": 400, "width": 400, "p": 1.0}},
                {"class_name": "albumentations.Rotate", "params": {"limit": 10, "p": 0.5}},  # To be tuned
                {"class_name": "albumentations.ColorJitter", "params": {"brightness": 0.1, "contrast": 0.1, "p": 0.1}},  # To be tuned
                {"class_name": "albumentations.HorizontalFlip", "params": {"p": 0.5}},  # To be tuned
                {"class_name": "albumentations.Normalize", "params": {}},
                {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}}
            ]
        },
        "valid": {
            "augs": [
                {"class_name": "albumentations.Resize", "params": {"height": 400, "width": 400, "p": 1.0}},
                {"class_name": "albumentations.Normalize", "params": {}},
                {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}}
            ]
        }
    },
    "test": {
        "folder_path": "None"  # Set to "None" (string) if no test data is available
    },
    "optuna": {
        "use_optuna": True,
        "n_trials": 3,   # Quick test: 3 trials
        "params": {
            "lr": {"min": 1e-5, "max": 1e-3, "type": "loguniform"},
            "batch_size": {"values": [4, 8], "type": "categorical"},
            "gradient_clip_val": {"min": 0.0, "max": 0.3, "type": "float"},
            "weight_decay": {"min": 0.0, "max": 0.01, "type": "float"},
            "rotation_limit": {"min": 5, "max": 15, "type": "int"},
            "color_jitter_strength": {"min": 0.1, "max": 0.3, "type": "float"},
            "horizontal_flip_prob": {"min": 0.3, "max": 0.7, "type": "float"}
        }
    },
    "pretrained_ckpt": "None"  # Set a checkpoint path if continuing training from a saved model
}

cfg: DictConfig = OmegaConf.create(cfg_dict)
