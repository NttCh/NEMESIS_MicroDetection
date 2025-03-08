"""
main.py

Main entry point for training and inference.
"""

import os
import datetime
import time
import copy
import optuna
from tqdm import tqdm
from omegaconf import OmegaConf
import pytorch_lightning as pl
from config import cfg
from train import (train_stage, train_with_cross_validation,
                   repeated_cross_validation, continue_training, objective_stage)
from inference import evaluate_model, display_sample_predictions, predict_test_folder
from utils import set_seed, thai_time

# Set the global base save directory
BASE_SAVE_DIR = os.path.join(cfg.general.save_dir, thai_time().strftime("%Y%m%d"))
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

def main():
    set_seed(cfg.training.seed)
    best_model_folder = os.path.join(BASE_SAVE_DIR, "best_model")
    os.makedirs(best_model_folder, exist_ok=True)
    
    pretrained_ckpt = cfg.get("pretrained_ckpt", "None")
    detection_csv = cfg.data.detection_csv
    model = None

    if cfg.stage_detection:
        print("[Main] Training Stage 1: Detection (binary classification)")
        
        # Branch based on CV and optuna settings
        if cfg.training.cross_validation and cfg.optuna.use_optuna:
            def objective(trial):
                trial_cfg = copy.deepcopy(cfg)
                # Set tunable optimizer and data parameters dynamically
                for param_name, param_info in trial_cfg.optuna.params.items():
                    if param_name in ["rotation_limit", "color_jitter_strength", "horizontal_flip_prob"]:
                        continue
                    if param_info["type"] == "loguniform":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_float(param_name, param_info["min"], param_info["max"], log=True)
                    elif param_info["type"] == "categorical":
                        trial_cfg.data.batch_size = trial.suggest_categorical(param_name, param_info["values"])
                    elif param_info["type"] == "float":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_float(param_name, param_info["min"], param_info["max"])
                    elif param_info["type"] == "int":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_int(param_name, param_info["min"], param_info["max"])
                # Update augmentation parameters
                rotation_limit = trial.suggest_int("rotation_limit", cfg.optuna.params["rotation_limit"]["min"], cfg.optuna.params["rotation_limit"]["max"])
                color_jitter_strength = trial.suggest_float("color_jitter_strength", cfg.optuna.params["color_jitter_strength"]["min"], cfg.optuna.params["color_jitter_strength"]["max"])
                horizontal_flip_prob = trial.suggest_float("horizontal_flip_prob", cfg.optuna.params["horizontal_flip_prob"]["min"], cfg.optuna.params["horizontal_flip_prob"]["max"])
                for aug in trial_cfg.augmentation.train.augs:
                    if aug["class_name"] == "albumentations.Rotate":
                        aug["params"]["limit"] = rotation_limit
                    elif aug["class_name"] == "albumentations.ColorJitter":
                        aug["params"]["brightness"] = color_jitter_strength
                        aug["params"]["contrast"] = color_jitter_strength
                    elif aug["class_name"] == "albumentations.HorizontalFlip":
                        aug["params"]["p"] = horizontal_flip_prob
                trial_cfg.trainer.max_epochs = trial_cfg.training.tuning_epochs_detection
                scores = []
                from train import train_with_cross_validation
                for _ in range(trial_cfg.training.repeated_cv):
                    _, score = train_with_cross_validation(trial_cfg, detection_csv, 2, "detection")
                    scores.append(score)
                return float(sum(scores) / len(scores))
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=True)
            print("[Optuna] Best trial:", study.best_trial)
            trials_df = study.trials_dataframe()
            eval_folder = os.path.join(BASE_SAVE_DIR, "eval")
            os.makedirs(eval_folder, exist_ok=True)
            optuna_excel_path = os.path.join(eval_folder, "optuna_trials.xlsx")
            trials_df.to_excel(optuna_excel_path, index=False)
            print(f"[Optuna] Results saved to {optuna_excel_path}")
            # Save hyperparameter tuning visualizations
            import optuna.visualization as vis
            vis.plot_optimization_history(study).write_image(os.path.join(eval_folder, "opt_history.png"))
            vis.plot_param_importances(study).write_image(os.path.join(eval_folder, "param_importance.png"))
            vis.plot_slice(study, params=list(study.best_trial.params.keys())).write_image(os.path.join(eval_folder, "slice.png"))
            # Load best hyperparameters into cfg
            best_params = study.best_trial.params
            for k, v in best_params.items():
                if k == "batch_size":
                    cfg.data.batch_size = v
                elif k in ["lr", "weight_decay", "gradient_clip_val"]:
                    cfg.optimizer.params[k] = v
                elif k == "rotation_limit":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.Rotate":
                            aug["params"]["limit"] = v
                elif k == "color_jitter_strength":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.ColorJitter":
                            aug["params"]["brightness"] = v
                            aug["params"]["contrast"] = v
                elif k == "horizontal_flip_prob":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.HorizontalFlip":
                            aug["params"]["p"] = v
            from train import repeated_cross_validation
            detection_model, final_score = repeated_cross_validation(cfg, detection_csv, 2, "detection", repeats=cfg.training.repeated_cv)
            print(f"[Main] Final repeated CV average score: {final_score:.4f}")
        
        elif cfg.training.cross_validation and not cfg.optuna.use_optuna:
            from train import train_with_cross_validation
            detection_model, _ = train_with_cross_validation(cfg, detection_csv, 2, "detection")
        
        elif not cfg.training.cross_validation and cfg.optuna.use_optuna:
            def objective(trial):
                trial_cfg = copy.deepcopy(cfg)
                for param_name, param_info in trial_cfg.optuna.params.items():
                    if param_info["type"] == "loguniform":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_float(param_name, param_info["min"], param_info["max"], log=True)
                    elif param_info["type"] == "categorical":
                        trial_cfg.data.batch_size = trial.suggest_categorical(param_name, param_info["values"])
                    elif param_info["type"] == "float":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_float(param_name, param_info["min"], param_info["max"])
                    elif param_info["type"] == "int":
                        trial_cfg.optimizer.params[param_name] = trial.suggest_int(param_name, param_info["min"], param_info["max"])
                trial_cfg.trainer.max_epochs = trial_cfg.training.tuning_epochs_detection
                from train import train_stage
                model, val_acc = train_stage(trial_cfg, detection_csv, 2, "detection", trial=trial, suppress_metrics=True)
                return val_acc.item() if val_acc else 0.0
            import optuna
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=True)
            best_params = study.best_trial.params
            for k, v in best_params.items():
                if k == "batch_size":
                    cfg.data.batch_size = v
                elif k in ["lr", "weight_decay", "gradient_clip_val"]:
                    cfg.optimizer.params[k] = v
                elif k == "rotation_limit":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.Rotate":
                            aug["params"]["limit"] = v
                elif k == "color_jitter_strength":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.ColorJitter":
                            aug["params"]["brightness"] = v
                            aug["params"]["contrast"] = v
                elif k == "horizontal_flip_prob":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.HorizontalFlip":
                            aug["params"]["p"] = v
            from train import train_stage
            detection_model, detection_val_acc = train_stage(cfg, detection_csv, 2, "detection", trial_number=1)
            print(f"[Main] Final tuned val_acc: {detection_val_acc:.4f}")
        
        else:
            from train import train_stage
            detection_model, detection_val_acc = train_stage(cfg, detection_csv, 2, "detection", trial_number=None)
        
        # Save best model checkpoint
        best_ckpt = os.path.join(best_model_folder, "best_detection.ckpt")
        import torch
        torch.save(detection_model.state_dict(), best_ckpt)
        print(f"[Main] Saved detection checkpoint to {best_ckpt}")
        
        # Continue training if desired
        from train import continue_training
        detection_model = continue_training(detection_model, cfg, detection_csv, 2, "detection")
        
        # Evaluate and display predictions
        from inference import evaluate_model, display_sample_predictions, predict_test_folder
        evaluate_model(detection_model, detection_csv, cfg, stage="Detection")
        full_df = pd.read_csv(detection_csv)
        from sklearn.model_selection import train_test_split
        _, valid_df = train_test_split(full_df, test_size=cfg.data.valid_split, random_state=cfg.training.seed,
                                        stratify=full_df[cfg.data.label_col])
        from albumentations import Compose
        valid_transforms = Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
        from data import PatchClassificationDataset
        valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
        display_sample_predictions(detection_model, valid_dataset, num_samples=4)
        if str(cfg.test.folder_path).lower() != "none":
            test_folder = cfg.test.folder_path
            output_csv = os.path.join(os.getcwd(), "test_predictions.csv")
            valid_transform = Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
            predict_test_folder(detection_model, test_folder, valid_transform, output_csv,
                                print_results=True, model_path=pretrained_ckpt)
    
    print("[Main] Training finished. Best model is saved in:", best_model_folder)


if __name__ == "__main__":
    main()
