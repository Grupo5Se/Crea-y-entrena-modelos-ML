from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import joblib
import numpy as np
from PIL import Image, ImageEnhance
from skimage.feature import hog

from .config import DEFAULT_IMAGE_SIZE, MODEL_OUTPUT_DIR

MODEL_FILE = MODEL_OUTPUT_DIR / 'road_damage_model.joblib'
SUMMARY_FILE = MODEL_OUTPUT_DIR / 'training_summary.json'


def load_image(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    image = Image.open(image_path).convert('L').resize((image_size, image_size))
    return np.array(image, dtype=np.uint8)


def extract_hog_features(image_array: np.ndarray) -> np.ndarray:
    return hog(
        image_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True,
    )


def build_feature_vector(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    return extract_hog_features(load_image(image_path, image_size=image_size))


def create_augmented_arrays(image_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> List[np.ndarray]:
    base = Image.open(image_path).convert('L').resize((image_size, image_size))
    variants = [
        base,
        base.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        base.rotate(7),
        base.rotate(-7),
        ImageEnhance.Contrast(base).enhance(1.35),
        ImageEnhance.Brightness(base).enhance(1.15),
    ]
    return [np.array(img, dtype=np.uint8) for img in variants]


def load_model_bundle() -> Dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError('No se encontró un modelo entrenado.')
    return joblib.load(MODEL_FILE)


def predict_image(image_path: Path) -> Dict:
    bundle = load_model_bundle()
    model = bundle['model']
    class_names = bundle['class_names']
    image_size = bundle.get('image_size', DEFAULT_IMAGE_SIZE)
    vector = build_feature_vector(image_path, image_size=image_size).reshape(1, -1)
    probabilities = model.predict_proba(vector)[0]
    pred_index = int(np.argmax(probabilities))
    label = class_names[pred_index]
    confidence = float(probabilities[pred_index])
    scores = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    return {
        'label': label,
        'confidence': confidence,
        'scores': scores,
        'model_path': str(MODEL_FILE),
    }


def read_training_summary() -> Dict:
    if not SUMMARY_FILE.exists():
        return {}
    return json.loads(SUMMARY_FILE.read_text(encoding='utf-8'))


def dataset_counts(raw_dataset_dir: Path) -> Dict[str, int]:
    result: Dict[str, int] = {}
    if not raw_dataset_dir.exists():
        return result
    for label_dir in sorted([p for p in raw_dataset_dir.iterdir() if p.is_dir()]):
        result[label_dir.name] = sum(1 for p in label_dir.iterdir() if p.is_file())
    return result
