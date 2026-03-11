from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from src.paths import CLASS_NAMES_DIR, MODELS_DIR


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Tiny model to verify save/load works.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model_path = MODELS_DIR / "smoketest_model.keras"
    model.save(model_path)

    class_names = ["class_a", "class_b", "class_c"]
    class_names_path = CLASS_NAMES_DIR / "class_names.json"
    class_names_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded_model = tf.keras.models.load_model(model_path)
    loaded_class_names = json.loads(class_names_path.read_text(encoding="utf-8"))

    print("OK:", model_path.name, "loaded; classes:", loaded_class_names)


if __name__ == "__main__":
    main()

