import tensorflow as tf
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_efficientnet_v2.keras"
OUT_PATH   = BASE_DIR / "models" / "plant_species_model.tflite"
NAMES_SRC  = BASE_DIR / "class_names" / "class_names.json"
NAMES_DST  = BASE_DIR / "models" / "class_names.json"

print("Model yukleniyor...")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Model yuklendi.")

print("TFLite'a donusturuluyor...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # boyutu kuculttur (~4x)
tflite_model = converter.convert()

with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Kaydedildi: {OUT_PATH}")
print(f"Boyut: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

# class_names.json'u da models/ klasorune kopyala (uygulamaya kolay eklemek icin)
import shutil
shutil.copy(NAMES_SRC, NAMES_DST)
print(f"class_names.json kopyalandi: {NAMES_DST}")
print("\nMobil uygulamaya eklenecek 2 dosya:")
print(f"  {OUT_PATH}")
print(f"  {NAMES_DST}")
