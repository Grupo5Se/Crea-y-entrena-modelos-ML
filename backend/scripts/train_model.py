from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from backend.app.config import DEFAULT_IMAGE_SIZE, MODEL_OUTPUT_DIR, RAW_DATASET_DIR
from backend.app.ml_utils import create_augmented_arrays, dataset_counts, extract_hog_features

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MODEL_FILE = MODEL_OUTPUT_DIR / 'road_damage_model.joblib'
SUMMARY_FILE = MODEL_OUTPUT_DIR / 'training_summary.json'


def build_dataset():
    X = []
    y = []
    source_counts = {}

    for label_dir in sorted([p for p in RAW_DATASET_DIR.iterdir() if p.is_dir()]):
        label = label_dir.name
        files = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
        source_counts[label] = len(files)
        for file_path in files:
            for arr in create_augmented_arrays(file_path, image_size=DEFAULT_IMAGE_SIZE):
                X.append(extract_hog_features(arr))
                y.append(label)

    if len(set(y)) < 2:
        raise ValueError('Se necesitan al menos 2 clases para entrenar el modelo.')

    return np.array(X), np.array(y), source_counts


def main() -> None:
    print('=== ENTRENAMIENTO DETECTOR VIAL ===')
    print(f'Fecha: {datetime.now().isoformat(timespec="seconds")}')
    print(f'Ruta dataset: {RAW_DATASET_DIR}')

    X, y, source_counts = build_dataset()
    label_distribution = Counter(y)
    print(f'Total muestras con aumento: {len(X)}')
    print(f'Distribución: {dict(label_distribution)}')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced', gamma='scale', C=2.0)),
    ])

    print('Entrenando modelo SVM con HOG...')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred, labels=sorted(set(y))).tolist()
    class_names = list(pipeline.named_steps['svc'].classes_)

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': pipeline,
        'class_names': class_names,
        'image_size': DEFAULT_IMAGE_SIZE,
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'feature_extractor': 'HOG',
        'classifier': 'SVC',
    }, MODEL_FILE)

    summary = {
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'model_file': str(MODEL_FILE),
        'image_size': DEFAULT_IMAGE_SIZE,
        'feature_extractor': 'HOG',
        'classifier': 'SVC (RBF)',
        'accuracy': acc,
        'class_names': class_names,
        'source_counts': source_counts,
        'augmented_distribution': dict(label_distribution),
        'confusion_matrix': matrix,
        'classification_report': report,
        'dataset_counts_live': dataset_counts(RAW_DATASET_DIR),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Accuracy: {acc:.4f}')
    print('Clases:', class_names)
    print('Modelo guardado en:', MODEL_FILE)
    print('Resumen guardado en:', SUMMARY_FILE)
    print('=== ENTRENAMIENTO FINALIZADO ===')


if __name__ == '__main__':
    main()
