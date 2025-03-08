"""
data.py

Defines the custom dataset(s) and any data-loading utilities.
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Any, Optional, Tuple
import cv2
import numpy as np


class PatchClassificationDataset(Dataset):
    """Custom dataset for patch classification tasks."""
    def __init__(self, data: Any, image_dir: str, transforms: Optional[Any] = None, image_col: str = "filename") -> None:
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data
        valid_rows = []
        for idx, row in df.iterrows():
            primary_path = os.path.join(image_dir, row[image_col])
            alternative_folder = image_dir.replace("Fa", "test")
            alternative_path = os.path.join(alternative_folder, row[image_col])
            if os.path.exists(primary_path) or os.path.exists(alternative_path):
                valid_rows.append(row)
            else:
                print(f"Warning: Image not found for row {idx}: {primary_path} or {alternative_path}. Skipping.")
        self.df = pd.DataFrame(valid_rows)
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_col = image_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        row = self.df.iloc[idx]
        primary_path = os.path.join(self.image_dir, row[self.image_col])
        if os.path.exists(primary_path):
            image_path = primary_path
        else:
            alternative_folder = self.image_dir.replace("Fa", "test")
            alternative_path = os.path.join(alternative_folder, row[self.image_col])
            if os.path.exists(alternative_path):
                image_path = alternative_path
                print(f"Using alternative image path: {image_path}")
            else:
                raise FileNotFoundError(f"Image not found: {primary_path} or {alternative_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        label = int(row["label"])
        return image, label
