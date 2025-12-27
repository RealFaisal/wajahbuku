from typing import List, Dict, Union
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import os

# === ADJUSTABLE SETTINGS ===
# Model trained with this emotion ordering:
# 0: angry, 1: disgust, 2: fear, 3: happy, 4: neutral, 5: sad, 6: surprise
EMOTION_LABELS: List[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Image size & grayscale settings — adjust to match training preprocessing
IMG_SIZE: int = 224      # if trained with 48x48, set to 48
GRAYSCALE: bool = False  # set True if model expects 1 channel
# Normalize values (if model trained with ImageNet norm, keep these; otherwise adjust)
NORM_MEAN = [0.485, 0.456, 0.406] if not GRAYSCALE else [0.5]
NORM_STD = [0.229, 0.224, 0.225] if not GRAYSCALE else [0.5]
# ============================


class EmotionPredictor:
    def __init__(self, model_path: str, device: Union[str, torch.device] = None):
        self.model_path = model_path
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self._build_transform()

    def _build_transform(self):
        transforms = []
        if GRAYSCALE:
            transforms.append(T.Grayscale(num_output_channels=1))
        transforms.extend([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        return T.Compose(transforms)

    def _load_model(self, path: str):
        """
        Loads a PyTorch model. Here we assume the checkpoint contains state_dict.
        If the checkpoint is a full model object, you may need to adjust.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        # You should replace the model architecture creation with the exact
        # architecture used during training. For a generic resnet50 backbone:
        try:
            from torchvision import models
            model = models.resnet50(weights=None)  # weights=None instead of pretrained=False
            # adjust final layer size to 7 classes (FER2013)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, len(EMOTION_LABELS)))
            state = torch.load(path, map_location="cpu")
            # state could be a state_dict or a checkpoint with 'state_dict' key
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # sometimes keys have 'module.' prefix from DataParallel
            new_state = {}
            for k, v in state.items():
                new_k = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_k] = v
            model.load_state_dict(new_state)
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def predict(self, pil_image: Image.Image) -> Dict:
        """
        Input: PIL.Image (face-cropped recommended)
        Returns: {label: str, score: float, probs: [float]}
        """
        img = pil_image.convert("RGB") if not GRAYSCALE else pil_image.convert("L")
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # shape: (1, C, H, W)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
        top_idx = int(np.argmax(probs))
        return {
            "label": EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else str(top_idx),
            "score": float(probs[top_idx]),
            "probs": [float(p) for p in probs]
        }
