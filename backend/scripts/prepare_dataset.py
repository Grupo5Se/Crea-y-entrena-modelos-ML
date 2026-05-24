from __future__ import annotations

from pathlib import Path
import shutil

from backend.app.config import RAW_DATASET_DIR


def infer_label(file_name: str) -> str | None:
    lowered = file_name.lower()
    if 'bache' in lowered:
        return 'bache'
    if 'fisura' in lowered:
        return 'fisura'
    return None


def main() -> None:
    base = RAW_DATASET_DIR
    unlabeled = base / '_por_clasificar'
    if not unlabeled.exists():
        print('No existe la carpeta _por_clasificar. No hay nada que mover.')
        return

    moved = 0
    for file in unlabeled.iterdir():
        if not file.is_file():
            continue
        label = infer_label(file.name)
        if not label:
            print(f'No se pudo inferir la clase para: {file.name}')
            continue
        target = base / label / file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(target))
        moved += 1
        print(f'Movido: {file.name} -> {label}')

    print(f'Total movidos: {moved}')


if __name__ == '__main__':
    main()
