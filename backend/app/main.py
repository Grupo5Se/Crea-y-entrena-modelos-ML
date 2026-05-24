from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
import shutil
import subprocess
import sys
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    APP_NAME,
    API_HOST,
    API_PORT,
    FRONTEND_ORIGINS,
    LOG_DIR,
    MODEL_OUTPUT_DIR,
    RAW_DATASET_DIR,
    UPLOAD_PREDICT_DIR,
    UPLOAD_TRAIN_DIR,
)
from .ml_utils import dataset_counts, predict_image, read_training_summary

app = FastAPI(title=APP_NAME, version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

TRAIN_PROCESS: subprocess.Popen | None = None
TRAIN_LOG_FILE = LOG_DIR / 'training.log'


def _sanitize_filename(filename: str) -> str:
    cleaned = filename.replace('..', '').replace('/', '_').replace('\\', '_').strip()
    return cleaned or f'image_{uuid.uuid4().hex}.jpg'


@app.get('/api/health')
def health():
    return {
        'status': 'ok',
        'app': APP_NAME,
        'host': API_HOST,
        'port': API_PORT,
        'time': datetime.now().isoformat(timespec='seconds'),
    }


@app.get('/api/stats')
def stats():
    summary = read_training_summary()
    return {
        'dataset_counts': dataset_counts(RAW_DATASET_DIR),
        'has_model': (MODEL_OUTPUT_DIR / 'road_damage_model.joblib').exists(),
        'training_summary': summary,
    }


@app.post('/api/upload/training')
async def upload_training_images(label: str = Form(...), files: List[UploadFile] = File(...)):
    label = label.strip().lower()
    if label not in {'bache', 'fisura'}:
        raise HTTPException(status_code=400, detail='La etiqueta debe ser bache o fisura.')

    target_dir = RAW_DATASET_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for file in files:
        filename = _sanitize_filename(file.filename)
        suffix = Path(filename).suffix.lower() or '.jpg'
        final_name = f'{Path(filename).stem}_{uuid.uuid4().hex[:8]}{suffix}'
        destination = target_dir / final_name
        with destination.open('wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(final_name)

    return {
        'message': 'Imágenes guardadas para entrenamiento.',
        'label': label,
        'saved_files': saved_files,
        'dataset_counts': dataset_counts(RAW_DATASET_DIR),
    }


@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    filename = _sanitize_filename(file.filename)
    suffix = Path(filename).suffix.lower() or '.jpg'
    stored_name = f'predict_{uuid.uuid4().hex[:8]}{suffix}'
    destination = UPLOAD_PREDICT_DIR / stored_name
    with destination.open('wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_image(destination)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Error al procesar la imagen: {exc}') from exc

    result['filename'] = stored_name
    return result


@app.post('/api/train/start')
def start_training():
    global TRAIN_PROCESS
    if TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None:
        return {'status': 'running', 'message': 'Ya hay un entrenamiento en ejecución.'}

    TRAIN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_FILE.write_text('', encoding='utf-8')
    command = [sys.executable, '-m', 'backend.scripts.train_model']
    with TRAIN_LOG_FILE.open('a', encoding='utf-8') as log_file:
        TRAIN_PROCESS = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    return {'status': 'started', 'message': 'Entrenamiento iniciado correctamente.'}


@app.get('/api/train/status')
def training_status():
    global TRAIN_PROCESS
    running = TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None
    exit_code = None if running or TRAIN_PROCESS is None else TRAIN_PROCESS.poll()
    summary = read_training_summary()
    return {
        'running': running,
        'exit_code': exit_code,
        'log_file': str(TRAIN_LOG_FILE),
        'last_training': summary,
    }


@app.get('/api/train/logs')
def training_logs():
    if not TRAIN_LOG_FILE.exists():
        return {'logs': 'Aún no hay logs de entrenamiento.'}
    content = TRAIN_LOG_FILE.read_text(encoding='utf-8', errors='ignore')
    return {'logs': content[-12000:]}
