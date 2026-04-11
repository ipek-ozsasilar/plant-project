"""
TTA (Test Time Augmentation) ile tek görüntü tahmini.

Kullanım:
    python src/predict_tta.py goruntu.jpg
    python src/predict_tta.py goruntu.jpg --top 5
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

from src.paths import MODELS_DIR, CLASS_NAMES_DIR

IMG_SIZE = (224, 224)
MODEL_PATH = MODELS_DIR / "best_finetuned_model_v2.keras"
NAMES_PATH  = CLASS_NAMES_DIR / "class_names.json"
ID_MAP_PATH = CLASS_NAMES_DIR / "plantnet_species_id_map.json"


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)          # (224, 224, 3)
    return arr[np.newaxis]                          # (1, 224, 224, 3)


def tta_augments(img: np.ndarray) -> list[np.ndarray]:
    """8 farklı dönüşüm üretir: orijinal + flip + 6 rot kombinasyonu."""
    h_flip = img[:, :, ::-1, :]                    # yatay flip
    v_flip = img[:, ::-1, :, :]                    # dikey flip

    variants = [img, h_flip, v_flip]
    for k in [1, 2, 3]:                            # 90, 180, 270 derece
        variants.append(np.rot90(img[0], k=k)[np.newaxis].copy())
        variants.append(np.rot90(h_flip[0], k=k)[np.newaxis].copy())

    return variants                                 # 8 varyant


def predict_tta(model, img: np.ndarray) -> np.ndarray:
    """TTA uygulayarak ortalama olasılık vektörü döner."""
    variants = tta_augments(img)
    preds = [model.predict(v, verbose=0) for v in variants]
    return np.mean(preds, axis=0)[0]               # (num_classes,)


def display_name(class_name: str, id_map: dict) -> str:
    if class_name.startswith("plantnet__"):
        pid = class_name.removeprefix("plantnet__")
        return id_map.get(pid, class_name)
    return class_name.removeprefix("leafsnap__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Tahmin edilecek görüntü dosyası")
    parser.add_argument("--top", type=int, default=3, help="Kaç sonuç gösterilsin (varsayılan: 3)")
    parser.add_argument("--no-tta", action="store_true", help="TTA'yı devre dışı bırak")
    args = parser.parse_args()

    print("Model yükleniyor...")
    model = keras.models.load_model(str(MODEL_PATH))

    with open(NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    id_map = {}
    if ID_MAP_PATH.exists():
        with open(ID_MAP_PATH, encoding="utf-8") as f:
            id_map = json.load(f)

    img = load_image(args.image)

    if args.no_tta:
        probs = model.predict(img, verbose=0)[0]
        label = "TTA kapalı"
    else:
        probs = predict_tta(model, img)
        label = "TTA açık (8 varyant)"

    top_idx = np.argsort(probs)[::-1][: args.top]

    print(f"\n[{label}] — {args.image}")
    print("-" * 45)
    for rank, idx in enumerate(top_idx, 1):
        name = display_name(class_names[idx], id_map)
        print(f"  {rank}. {name:<35} {probs[idx]*100:.1f}%")


if __name__ == "__main__":
    main()
