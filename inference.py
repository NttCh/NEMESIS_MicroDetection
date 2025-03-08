"""
inference.py

Contains functions for evaluation and test predictions.
"""

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from data import PatchClassificationDataset
from utils import load_obj

def evaluate_model(model, csv_path: str, cfg, stage: str) -> None:
    """
    Evaluate the model on the validation split and plot the confusion matrix.
    """
    full_df = pd.read_csv(csv_path)
    from sklearn.model_selection import train_test_split
    _, valid_df = train_test_split(full_df, test_size=cfg.data.valid_split, random_state=cfg.training.seed,
                                   stratify=full_df[cfg.data.label_col])
    from albumentations import Compose
    valid_transforms = Compose([load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs])
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {stage}")
    eval_folder = os.path.join(cfg.general.save_dir, "eval")
    os.makedirs(eval_folder, exist_ok=True)
    cm_save_path = os.path.join(eval_folder, "confusion_matrix.png")
    plt.savefig(cm_save_path)
    print(f"[Evaluate] Saved confusion matrix plot to {cm_save_path}")
    plt.show()
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))


def display_sample_predictions(model, dataset, num_samples: int = 4) -> None:
    """
    Display a grid of sample predictions.
    """
    if len(dataset) == 0:
        print("Dataset is empty, cannot show sample predictions.")
        return
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images, true_labels, pred_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for idx in indices:
            image, label = dataset[idx]
            if not hasattr(image, "shape"):
                image = image.permute(2, 0, 1)
            input_tensor = image.unsqueeze(0).to(model.device)
            logits = model(input_tensor)
            pred = logits.argmax(dim=1).item()
            images.append(image.cpu().permute(1, 2, 0).numpy())
            true_labels.append(label)
            pred_labels.append(pred)
    model.train()
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        axs[i].imshow(images[i].astype(np.uint8))
        axs[i].set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
        axs[i].axis("off")
    eval_folder = os.path.join(os.path.dirname(model.logger.log_dir), "eval")
    os.makedirs(eval_folder, exist_ok=True)
    sample_plot_path = os.path.join(eval_folder, "sample_predictions.png")
    plt.savefig(sample_plot_path)
    print(f"[Display] Saved sample predictions plot to {sample_plot_path}")
    plt.show()


def predict_test_folder(model, test_folder: str, transform, output_csv: str,
                        print_results: bool = True, model_path: Optional[str] = None) -> None:
    """
    Predict on test images from a folder and save the results.
    """
    if test_folder is None or str(test_folder).lower() == "none":
        print("No test folder provided. Skipping test predictions.")
        return
    if model_path is not None and model_path.lower() != "none":
        print(f"Loading model checkpoint from {model_path}")
        state_dict = torch.load(model_path, map_location=model.device)
        model.load_state_dict(state_dict)
    image_files = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))
    if not image_files:
        print("No image files found in test folder. Skipping test predictions.")
        return
    # Sort files (assuming filename contains a number, customize as needed)
    image_files = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
    predictions = []
    model.eval()
    with torch.no_grad():
        for file in image_files:
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read {file}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            augmented = transform(image=image)
            image_tensor = augmented["image"]
            if not hasattr(image_tensor, "shape"):
                image_tensor = torch.tensor(image_tensor)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(model.device)
            logits = model(image_tensor)
            pred = logits.argmax(dim=1).item()
            predictions.append({"filename": file, "predicted_label": pred})
            if print_results:
                print(f"File: {file} -> Predicted Label: {pred}")
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
