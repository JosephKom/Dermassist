"""Shared ResNet18 inference for the DermaAssist application."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = (
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
)
MODEL_PATH = Path(__file__).parent / "skin-disease-detection-main1" / "best_model.pth"
IMAGE_SIZE = 224


def _build_model() -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def _state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("best_model.pth does not contain a model state dictionary")

    for key in ("state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            checkpoint = nested
            break

    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ValueError("best_model.pth contains no tensor weights")
    return state_dict


class Predictor:
    """Load the classifier once and return probabilities for one image."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = _build_model()
        self.model.load_state_dict(_state_dict(checkpoint))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict[str, float]:
        """Return one probability for each HAM10000 class."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(image_tensor), dim=1)[0]

        return {
            class_name: float(probability)
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        }


_predictor: Predictor | None = None


def predict(image: Image.Image) -> dict[str, float]:
    """Predict an image using the process-wide shared model instance."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor.predict(image)
