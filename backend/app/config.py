from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
import os

BASE_FILE = Path(__file__).resolve()
BACKEND_DIR = BASE_FILE.parents[1]
PROJECT_DIR = BASE_FILE.parents[2]
ENV_PATH = BACKEND_DIR / '.env'
load_dotenv(ENV_PATH)


def _resolve_path(value: str, fallback: str) -> Path:
    raw = value or fallback
    path = Path(raw)
    if not path.is_absolute():
        path = (BACKEND_DIR / path).resolve()
    return path


APP_NAME = os.getenv('APP_NAME', 'Detector Vial ML')
API_HOST = os.getenv('API_HOST', '127.0.0.1')
API_PORT = int(os.getenv('API_PORT', '8000'))
FRONTEND_ORIGINS = [item.strip() for item in os.getenv('FRONTEND_ORIGINS', 'http://localhost,http://127.0.0.1').split(',') if item.strip()]
DEFAULT_IMAGE_SIZE = int(os.getenv('DEFAULT_IMAGE_SIZE', '128'))
RAW_DATASET_DIR = _resolve_path(os.getenv('RAW_DATASET_DIR', '../data/raw'), '../data/raw')
UPLOAD_TRAIN_DIR = _resolve_path(os.getenv('UPLOAD_TRAIN_DIR', '../data/uploads/training'), '../data/uploads/training')
UPLOAD_PREDICT_DIR = _resolve_path(os.getenv('UPLOAD_PREDICT_DIR', '../data/uploads/predictions'), '../data/uploads/predictions')
MODEL_OUTPUT_DIR = _resolve_path(os.getenv('MODEL_OUTPUT_DIR', '../models'), '../models')
LOG_DIR = _resolve_path(os.getenv('LOG_DIR', '../logs'), '../logs')

for path in [RAW_DATASET_DIR, UPLOAD_TRAIN_DIR, UPLOAD_PREDICT_DIR, MODEL_OUTPUT_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)
