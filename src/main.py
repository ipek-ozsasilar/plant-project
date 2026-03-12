# -*- coding: utf-8 -*-
# Bu dosya egitilmis_model.ipynb dosyasindan Python script formatina donusturuldu.
# Notebook hucreleri, siralari korunarak # %% ayiraclari ile eklendi.

# %% Cell 0
# === EĞİTİM SONU KAYIT HÜCRESİ ===
from pathlib import Path
import json
from src.paths import MODELS_DIR, CLASS_NAMES_DIR, OUTPUTS_DIR

# klasörleri garanti et
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# 1) Modeli kaydet
model_path = MODELS_DIR / "plant_model_250classes_local.keras"
model.save(model_path)
print("Model kaydedildi  ->", model_path)

# 2) Class isimlerini kaydet
class_names_path = CLASS_NAMES_DIR / "class_names.json"
with open(class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print("Class names kaydedildi ->", class_names_path)

# 3) History (loss/acc geçmişi) kaydet
history_path = OUTPUTS_DIR / "history_plant_model_250classes_local.json"
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history.history, f, ensure_ascii=False, indent=2)
print("History kaydedildi ->", history_path)

# %% Cell 1
from tensorflow import keras
import json
from src.paths import MODELS_DIR, CLASS_NAMES_DIR

model = keras.models.load_model(MODELS_DIR / "plant_model_250classes_local.keras")

with open(CLASS_NAMES_DIR / "class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

# %% Cell 2
from pathlib import Path
import sys

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("src klasörü var mı?:", (PROJECT_ROOT / "src").exists())

# %% Cell 3
import random, shutil
from pathlib import Path
from src.paths import DATA_DIR

random.seed(42)

SRC_ROOT = DATA_DIR / "leafsnap_subset_50" / "images" / "field"
OUT_ROOT = DATA_DIR / "leafsnap_split"

print("SRC_ROOT:", SRC_ROOT)
print("SRC_ROOT exists:", SRC_ROOT.exists())

train_dir = OUT_ROOT / "train"
val_dir = OUT_ROOT / "val"
test_dir = OUT_ROOT / "test"

for d in [train_dir, val_dir, test_dir]:
    d.mkdir(parents=True, exist_ok=True)

class_dirs = [p for p in SRC_ROOT.iterdir() if p.is_dir()]
print("Kaç sınıf klasörü var:", len(class_dirs))

train_count = val_count = test_count = 0

for cdir in class_dirs:
    imgs = list(cdir.glob("*.jpg"))
    print("Sınıf:", cdir.name, "görüntü sayısı:", len(imgs))  # debug satırı

    random.shuffle(imgs)
    n = len(imgs)

    train_split = int(n * 0.7)
    val_split = int(n * 0.15)

    train_imgs = imgs[:train_split]
    val_imgs = imgs[train_split:train_split+val_split]
    test_imgs = imgs[train_split+val_split:]

    for img in train_imgs:
        dst = train_dir / cdir.name
        dst.mkdir(exist_ok=True)
        shutil.copy2(img, dst / img.name)
        train_count += 1

    for img in val_imgs:
        dst = val_dir / cdir.name
        dst.mkdir(exist_ok=True)
        shutil.copy2(img, dst / img.name)
        val_count += 1

    for img in test_imgs:
        dst = test_dir / cdir.name
        dst.mkdir(exist_ok=True)
        shutil.copy2(img, dst / img.name)
        test_count += 1

print("Train görüntü:", train_count)
print("Validation görüntü:", val_count)
print("Test görüntü:", test_count)
print("Toplam:", train_count + val_count + test_count)

# %% Cell 4
#images/field içindeki tüm sınıfları buldu, her sınıfta kaç foto var saydı, en çok/en az olanları yazdırdı.
from pathlib import Path
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("..") / "data" / "leafsnap-dataset" / "dataset"
FIELD_DIR = DATA_ROOT / "images" / "field"

print("FIELD_DIR var mı?:", FIELD_DIR.exists())
print("FIELD_DIR:", FIELD_DIR)

# sınıflar (klasör isimleri)
class_dirs = sorted([p for p in FIELD_DIR.iterdir() if p.is_dir()])
print("Toplam sınıf sayısı:", len(class_dirs))
print("İlk 10 sınıf:", [p.name for p in class_dirs[:10]])

# sınıf başına görüntü sayısı
counts = Counter()
for cdir in class_dirs:
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        n += len(list(cdir.glob(ext)))
    counts[cdir.name] = n

total_imgs = sum(counts.values())
print("Toplam görüntü:", total_imgs)

print("\nEn çok görüntü olan 10 sınıf:")
for name, n in counts.most_common(10):
    print(name, n)

print("\nEn az görüntü olan 10 sınıf:")
for name, n in counts.most_common()[-10:]:
    print(name, n)

# %% Cell 5
#Leafsnap field’den 50 sınıf seçti, her sınıftan en fazla 80 foto alarak 
#/kaggle/working/leafsnap_subset_50/... içine kopyaladı. yani subset oluşturma işlemini yaptık.
from pathlib import Path
from collections import Counter
import random, shutil

random.seed(42)

from pathlib import Path
from src.paths import DATA_DIR

DATA_ROOT = DATA_DIR / "leafsnap-dataset" / "dataset"
FIELD_DIR = DATA_ROOT / "images" / "field"

OUT_ROOT = DATA_DIR / "leafsnap_subset_50"
OUT_FIELD = OUT_ROOT / "images" / "field"

# temiz başla
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
OUT_FIELD.mkdir(parents=True, exist_ok=True)

# sınıf başına sayım
class_dirs = sorted([p for p in FIELD_DIR.iterdir() if p.is_dir()])
counts = Counter()

for cdir in class_dirs:
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        n += len(list(cdir.glob(ext)))
    counts[cdir.name] = n

min_images_per_class = 30
K = 50
cap_per_class = 80

eligible = [c for c, n in counts.items() if n >= min_images_per_class]
eligible_sorted = sorted(eligible, key=lambda c: counts[c], reverse=True)

if len(eligible_sorted) < K:
    raise ValueError(f"Yeterli sınıf yok. Eligible={len(eligible_sorted)} ama K={K}")

chosen_classes = eligible_sorted[:K]
print("Seçilen sınıf sayısı:", len(chosen_classes))
print("Seçilen ilk 10 sınıf:", chosen_classes[:10])
print("Seçilen sınıfların (orijinal) min/max görüntü sayısı:",
      min(counts[c] for c in chosen_classes), max(counts[c] for c in chosen_classes))

# KOPYALAMA
total_copied = 0
for cls in chosen_classes:
    src_dir = FIELD_DIR / cls
    dst_dir = OUT_FIELD / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend(list(src_dir.glob(ext)))

    random.shuffle(imgs)
    imgs = imgs[:min(cap_per_class, len(imgs))]

    for img_path in imgs:
        shutil.copy2(img_path, dst_dir / img_path.name)
    total_copied += len(imgs)

print("Subset oluşturuldu ✅")
print("OUT_ROOT:", OUT_ROOT)
print("Toplam kopyalanan görüntü:", total_copied)

# hızlı kontrol
subset_class_dirs = sorted([p for p in OUT_FIELD.iterdir() if p.is_dir()])
print("Subset sınıf sayısı:", len(subset_class_dirs))
print("Örnek subset sınıfları:", [p.name for p in subset_class_dirs[:10]])

# %% Cell 6
#50 sınıfın her birini aldı görüntüleri karıştırdı %70 train %15 val %15 test olarak böldü.
import random, shutil
from pathlib import Path
from src.paths import DATA_DIR

random.seed(42)

SRC_ROOT = DATA_DIR / "leafsnap_subset_50" / "images" / "field"
OUT_ROOT = DATA_DIR / "leafsnap_split"

print("SRC_ROOT:", SRC_ROOT)
print("SRC_ROOT exists:", SRC_ROOT.exists())

train_dir = OUT_ROOT / "train"
val_dir = OUT_ROOT / "val"
test_dir = OUT_ROOT / "test"

# klasörleri oluştur
for d in [train_dir, val_dir, test_dir]:
    d.mkdir(parents=True, exist_ok=True)

class_dirs = [p for p in SRC_ROOT.iterdir() if p.is_dir()]
print("Kaç sınıf klasörü var:", len(class_dirs))

train_count = 0
val_count = 0
test_count = 0

for cdir in class_dirs:
    imgs = list(cdir.glob("*.jpg"))
    random.shuffle(imgs)

    n = len(imgs)

    train_split = int(n * 0.7)
    val_split = int(n * 0.15)

    train_imgs = imgs[:train_split]
    val_imgs = imgs[train_split:train_split+val_split]
    test_imgs = imgs[train_split+val_split:]

    for img in train_imgs:
        dst = train_dir / cdir.name
        dst.mkdir(exist_ok=True)
        shutil.copy2(img, dst / img.name)
        train_count += 1

    for img in val_imgs:
        dst = val_dir / cdir.name
        dst.mkdir(exist_ok=True)
        shutil.copy2(img, dst / img.name)
        val_count += 1

    for img in test_imgs:
        dst = test_dir / cdir.name
        dst.mkdir(exist_ok=True)
        shutil.copy2(img, dst / img.name)
        test_count += 1

print("Train görüntü:", train_count)
print("Validation görüntü:", val_count)
print("Test görüntü:", test_count)
print("Toplam:", train_count + val_count + test_count)

# %% Cell 7
#MobileNetV2’yi hazır alacağız (önceden eğitilmiş) Gövdeyi donduracağız (freeze)
#Sadece en sona “bizim 50 sınıfımız için” yeni bir katman ekleyip eğiteceğiz
#image_dataset_from_directory: klasör yapısından otomatik dataset oluşturdu
#MobileNetV2(weights="imagenet"): hazır modeli indirdi base_model.trainable = False: dondurdu
#(yani öğrenmesini kapattı) Dense(num_classes): senin 50 sınıfın için yeni çıkış katmanı ekledi
#8 epoch eğitim yaptı ve test accuracy bastı
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from src.paths import DATA_DIR as ROOT_DATA_DIR  # <-- ek

DATA_DIR = ROOT_DATA_DIR / "leafsnap_split"      # <-- değişen satır
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Sınıf sayısı:", num_classes)
print("İlk 10 sınıf:", class_names[:10])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augmentation")

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

EPOCHS = 8
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# %% Cell 8
# Fine tuning başlıyor MobileNet’i açtı base_model.trainable = True
# İlk katmanları tekrar kapattı fine_tune_at = len(base_model.layers) - 30
#Yani sadece son 30 katman öğreniyor. Learning rate düşürdük 1e-3 → 1e-5 Çünkü artık ince ayar yapıyoruz.

base_model.trainable = True

# sadece üst katmanları aç
fine_tune_at = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

FINE_EPOCHS = 5

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_EPOCHS
)

test_loss, test_acc = model.evaluate(test_ds)

print("Fine tuned test accuracy:", test_acc)

# %% Cell 9
#Şimdi modelin nerede hata yaptığını, hangi türleri karıştırdığını görmemiz gerekiyor.Bu Confusion Matrix
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

y_true = []
y_pred = []

for images, labels in test_ds:
    
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)
    
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(18,18))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# %% Cell 10
import tensorflow as tf
from pathlib import Path
from src.paths import DATA_DIR as ROOT_DATA_DIR  # ek

DATA_DIR = ROOT_DATA_DIR / "leafsnap_split"      # değişti
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_dir = DATA_DIR / "test"

print("test_dir:", test_dir)
print("test_dir exists:", test_dir.exists())

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,  # önemli: sıralama bozulmasın
)

class_names = test_ds.class_names
print("Sınıf sayısı:", len(class_names))
print("İlk 10 sınıf:", class_names[:10])

# %% Cell 11
import numpy as np
import matplotlib.pyplot as plt
import random

# test setindeki tüm görüntüleri ve etiketleri RAM'e alalım (508 image -> rahat)
images = []
labels = []

for batch_imgs, batch_labels in test_ds:
    images.append(batch_imgs.numpy())
    labels.append(batch_labels.numpy())

images = np.concatenate(images, axis=0)
labels = np.concatenate(labels, axis=0)

print("Toplam test görüntü:", len(images))

# rastgele 12 örnek
random.seed(42)
idxs = random.sample(range(len(images)), 12)

sample_imgs = images[idxs]
sample_labels = labels[idxs].astype(int)

# model tahminleri
probs = model.predict(sample_imgs, verbose=0)  # model değişkenin adı "model" değilse söyle
pred_ids = np.argmax(probs, axis=1)
conf = np.max(probs, axis=1)

# çizim
plt.figure(figsize=(16, 10))
for i, idx in enumerate(idxs):
    true_id = sample_labels[i]
    pred_id = pred_ids[i]
    ok = (true_id == pred_id)

    true_name = class_names[true_id]
    pred_name = class_names[pred_id]
    c = conf[i] * 100

    plt.subplot(3, 4, i+1)
    plt.imshow(sample_imgs[i].astype("uint8"))
    plt.axis("off")
    plt.title(
        f"T: {true_name}\nP: {pred_name} ({c:.1f}%)",
        color=("green" if ok else "red"),
        fontsize=10
    )

plt.suptitle("LeafSnap Test Seti - Rastgele Örnek Tahminler (True vs Pred)", fontsize=14)
plt.tight_layout()
plt.show()

# %% Cell 12
#yeni datasetteki sınıf sayısını vs bul
from pathlib import Path
import os, glob

from src.paths import DATA_DIR

PLANTNET_ROOT = DATA_DIR / "plantnet_300K"

train_dir = PLANTNET_ROOT / "images_train"
val_dir   = PLANTNET_ROOT / "images_val"
test_dir  = PLANTNET_ROOT / "images_test"

print("PLANTNET_ROOT exists:", PLANTNET_ROOT.exists())
print("Train exists:", train_dir.exists())
print("Val exists:", val_dir.exists())
print("Test exists:", test_dir.exists())

# kaç sınıf klasörü var?
if train_dir.exists():
    classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    print("Train class count:", len(classes))
    print("First 10 classes:", classes[:10])

    # bir sınıftan örnek kaç jpg var?
    if classes:
        sample_class = train_dir / classes[0]
        jpg_count = len(list(sample_class.rglob("*.jpg")))
        print("Sample class:", classes[0], "jpg:", jpg_count)

# %% Cell 13
#subset cıkarıp workıng altına kopyalaıp eafsnap ıle bırlestırme ıslemı yapıyor
from pathlib import Path
import random
import shutil

random.seed(42)

# -------------------------
# YOLLAR
# -------------------------
from src.paths import DATA_DIR

LEAFSNAP_SPLIT = DATA_DIR / "leafsnap_split"
PLANTNET_ROOT  = DATA_DIR / "plantnet_300K"
PLANTNET_TRAIN = PLANTNET_ROOT / "images_train"
PLANTNET_VAL   = PLANTNET_ROOT / "images_val"
PLANTNET_TEST  = PLANTNET_ROOT / "images_test"

OUT = DATA_DIR / "combined_split"
OUT_TRAIN = OUT / "train"
OUT_VAL   = OUT / "val"
OUT_TEST  = OUT / "test"

# -------------------------
# TEMİZLE
# -------------------------
if OUT.exists():
    shutil.rmtree(OUT)

OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_VAL.mkdir(parents=True, exist_ok=True)
OUT_TEST.mkdir(parents=True, exist_ok=True)

# -------------------------
# PlantNet sınıf seçimi
# -------------------------
def count_images_in_classdir(cdir: Path):
    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        count += len(list(cdir.glob(ext)))
    return count

plantnet_class_dirs = [p for p in PLANTNET_TRAIN.iterdir() if p.is_dir()]
counts = {p.name: count_images_in_classdir(p) for p in plantnet_class_dirs}

MIN_PER_CLASS = 50
eligible = [cls for cls, n in counts.items() if n >= MIN_PER_CLASS]

N_SELECT = 200
selected_classes = random.sample(eligible, k=N_SELECT)

print("PlantNet total classes:", len(counts))
print("Eligible classes:", len(eligible))
print("Selected classes:", len(selected_classes))

# -------------------------
# KOPYALAMA (SINIRLI)
# -------------------------
MAX_TRAIN_PER_CLASS = 80
MAX_VAL_PER_CLASS   = 20
MAX_TEST_PER_CLASS  = 20

def copy_limited(src_root: Path, dst_root: Path, class_names, prefix: str, max_per_class: int):
    copied_files = 0
    copied_classes = 0

    for cls in class_names:
        src_cls = src_root / cls
        if not src_cls.exists():
            continue

        dst_cls = dst_root / f"{prefix}__{cls}"
        dst_cls.mkdir(parents=True, exist_ok=True)

        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs.extend(list(src_cls.glob(ext)))

        random.shuffle(imgs)
        imgs = imgs[:max_per_class]

        for img in imgs:
            shutil.copy2(img, dst_cls / img.name)
            copied_files += 1

        if len(imgs) > 0:
            copied_classes += 1

    return copied_files, copied_classes

# PlantNet kopyala
pn_train_files, pn_train_classes = copy_limited(PLANTNET_TRAIN, OUT_TRAIN, selected_classes, "plantnet", MAX_TRAIN_PER_CLASS)
pn_val_files,   pn_val_classes   = copy_limited(PLANTNET_VAL,   OUT_VAL,   selected_classes, "plantnet", MAX_VAL_PER_CLASS)
pn_test_files,  pn_test_classes  = copy_limited(PLANTNET_TEST,  OUT_TEST,  selected_classes, "plantnet", MAX_TEST_PER_CLASS)

print("\nPlantNet copied:")
print(" train files:", pn_train_files, "| classes:", pn_train_classes)
print(" val files  :", pn_val_files,   "| classes:", pn_val_classes)
print(" test files :", pn_test_files,  "| classes:", pn_test_classes)

# -------------------------
# Leafsnap'i TAM ekle
# -------------------------
def copy_all(src_root: Path, dst_root: Path, class_names, prefix: str):
    copied_files = 0
    copied_classes = 0

    for cls in class_names:
        src_cls = src_root / cls
        if not src_cls.exists():
            continue

        dst_cls = dst_root / f"{prefix}__{cls}"
        dst_cls.mkdir(parents=True, exist_ok=True)

        count_before = copied_files

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img in src_cls.glob(ext):
                shutil.copy2(img, dst_cls / img.name)
                copied_files += 1

        if copied_files > count_before:
            copied_classes += 1

    return copied_files, copied_classes

leaf_classes = [p.name for p in (LEAFSNAP_SPLIT / "train").iterdir() if p.is_dir()]

ls_train_files, ls_train_classes = copy_all(LEAFSNAP_SPLIT / "train", OUT_TRAIN, leaf_classes, "leafsnap")
ls_val_files,   ls_val_classes   = copy_all(LEAFSNAP_SPLIT / "val",   OUT_VAL,   leaf_classes, "leafsnap")
ls_test_files,  ls_test_classes  = copy_all(LEAFSNAP_SPLIT / "test",  OUT_TEST,  leaf_classes, "leafsnap")

print("\nLeafsnap copied:")
print(" train files:", ls_train_files, "| classes:", ls_train_classes)
print(" val files  :", ls_val_files,   "| classes:", ls_val_classes)
print(" test files :", ls_test_files,  "| classes:", ls_test_classes)

# -------------------------
# SON KONTROL
# -------------------------
def list_class_dirs(split_dir: Path):
    return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

train_classes = list_class_dirs(OUT_TRAIN)
val_classes   = list_class_dirs(OUT_VAL)
test_classes  = list_class_dirs(OUT_TEST)

print("\nCombined class counts:")
print(" train:", len(train_classes))
print(" val  :", len(val_classes))
print(" test :", len(test_classes))

print("\nSame class set across splits?:", set(train_classes) == set(val_classes) == set(test_classes))
print("OUT:", OUT)

# %% Cell 14
#Combined_split/train-val-test klasörlerini yükledi 250 sınıfı otomatik algıladı
#MobileNetV2 tabanını kullandı tabanı dondurdu (base_model.trainable = False) 
#Yeni 250 sınıflık çıkış katmanı kurdu modeli eğitti test doğruluğunu verdi
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from src.paths import DATA_DIR as ROOT_DATA_DIR  # <-- ek

# -------------------------
# DATASET YOLU
# -------------------------
DATA_DIR = ROOT_DATA_DIR / "combined_split"      # <-- değişti

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Toplam sınıf sayısı:", num_classes)
print("İlk 10 sınıf:", class_names[:10])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augmentation")

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

EPOCHS = 8

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# %% Cell 15
# Fine-tuning başlıyor

base_model.trainable = True

# Sadece son 30 katmanı aç
fine_tune_at = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Daha küçük learning rate ile tekrar derle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

FINE_EPOCHS = 5

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_EPOCHS
)

test_loss, test_acc = model.evaluate(test_ds)

print("Fine tuned test accuracy:", test_acc)

# %% Cell 16
# -------------------------
# DAHA IYI IYILESTIRME ICIN 2.FINE-TUNING
# -------------------------
base_model.trainable = True

fine_tune_at = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print("Toplam base_model katman sayısı:", len(base_model.layers))
print("Fine-tuning başlangıç katmanı:", fine_tune_at)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_finetuned_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
]

FINE_EPOCHS = 12

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_EPOCHS,
    callbacks=callbacks
)
test_loss, test_acc = model.evaluate(test_ds)
print("Fine-tuned test accuracy:", test_acc)

# %% Cell 17
# -------------------------
# DAHA IYI IYILESTIRME ICIN 3.FINE-TUNING (50 KATMAN AÇIK)
# -------------------------
base_model.trainable = True

# Son 50 katman trainable kalsın
fine_tune_at = len(base_model.layers) - 50

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print("Toplam base_model katman sayısı:", len(base_model.layers))
print("Fine-tuning başlangıç katmanı:", fine_tune_at)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # düşük LR
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_finetuned_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
]

FINE_EPOCHS = 12

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_EPOCHS,
    callbacks=callbacks
)

test_loss, test_acc = model.evaluate(test_ds)
print("Fine-tuned test accuracy:", test_acc)

# %% Cell 18
from pathlib import Path
from src.paths import DATA_DIR as ROOT_DATA_DIR

train_root = ROOT_DATA_DIR / "combined_split" / "train"

class_counts = {}
for class_dir in sorted(train_root.iterdir()):
    if not class_dir.is_dir():
        continue
    n = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        n += len(list(class_dir.glob(ext)))
    class_counts[class_dir.name] = n

print("Toplam sınıf:", len(class_counts))
print("Örnek:", list(class_counts.items())[:5])

# %% Cell 19
#class_weight hesapla (nadiren görülen sınıfları daha fazla ağırlıklandır)
#Bu, az görüntülü sınıfların loss içindeki ağırlığını arttırır → genelde val accuracy ve özellikle test accuracy tarafında daha dengeli sonuç verir.
import numpy as np

# class_counts: {'class_name': adet} sözlüğü
# class_names: train_ds.class_names listesi

total = sum(class_counts.values())
num_classes = len(class_names)

class_weights = {
    i: total / (num_classes * class_counts[name])
    for i, name in enumerate(class_names)
}

print("Örnek class_weights:", list(class_weights.items())[:10])

# %% Cell 20
#Aynı model üzerinde, sadece öğrenme oranını düşürüp hafif label smoothing ekleyerek birkaç epoch daha eğit:
# 50 katman açık hâliyle devam ediyoruz
#Daha düşük LR + label smoothing ile ek fine‑tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-6),
    loss="sparse_categorical_crossentropy",  # veya keras.losses.SparseCategoricalCrossentropy()
    metrics=["accuracy"]
)
EXTRA_FINE_EPOCHS = 10
history_fine_extra = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EXTRA_FINE_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)
test_loss2, test_acc2 = model.evaluate(test_ds)
print("Ek fine-tuning sonrası test accuracy:", test_acc2)

# %% Cell 21
import numpy as np
from sklearn.metrics import classification_report

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Val setinde GERÇEKTEN kullanılan sınıf id'leri
present_labels = sorted(set(y_true))
present_names = [class_names[i] for i in present_labels]

print(classification_report(
    y_true,
    y_pred,
    labels=present_labels,       # 226 sınıf id'si
    target_names=present_names   # 226 isim
))

# %% Cell 22
# === YENİ MODEL KURULUMU + İLK EĞİTİM ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
num_classes = len(class_names)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.10),
], name="augmentation")

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)  # daha güçlü regularization
outputs = layers.Dense(
    num_classes,
    activation="softmax",
    kernel_regularizer=keras.regularizers.l2(1e-4),
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 8
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

test_loss, test_acc = model.evaluate(test_ds)
print("Yeni model (sadece üst katman) test accuracy:", test_acc)

# %% Cell 23
# === SON 35 KATMAN İLE FINE-TUNING + CLASS_WEIGHT ===

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 35  # 50 yerine 35

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print("Toplam base_model katman sayısı:", len(base_model.layers))
print("Fine-tuning başlangıç katmanı:", fine_tune_at)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-6),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

FINE_EPOCHS = 12

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_EPOCHS,
    callbacks=callbacks,          # daha önce tanımladığın callbacks'i kullanıyor
    class_weight=class_weights,   # daha önce hesapladığın class_weights'i kullanıyor
)

test_loss_ft, test_acc_ft = model.evaluate(test_ds)
print("Yeni fine-tuned test accuracy:", test_acc_ft)

model.save("best_finetuned_model_v2.keras")
print("Model kaydedildi: best_finetuned_model_v2.keras")

