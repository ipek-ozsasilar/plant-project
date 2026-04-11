"""
Yeni, temiz egitim notebook'unu olusturur.
Calistirilinca notebooks/egitilmis_model.ipynb dosyasinin tum icerigi degistirilir.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "egitilmis_model.ipynb"

def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

# ===========================================================================
# HUCRE 0 — Baslik
# ===========================================================================
cells.append(md(
"""# Bitki Türü Tanıma Modeli — Eğitim Pipeline'ı
## Ne yapıyoruz?
- **93 bitki türü** için EfficientNetB3 modeli eğitiyoruz
- **PlantNet-300K** (200 görüntü/tür, mevcut) + **iNaturalist** (300 görüntü/tür, yeni indiriliyor) birleştiriyoruz
- iNaturalist: farklı açı, ışık, arka plan → model gerçek dünya fotoğraflarını tanıyacak
- **Agresif augmentation** ile veri çeşitliliği artırılıyor

## Hücre sırası
1. Imports + sabitler
2. Tür listesi (93 sınıf)
3. iNaturalist görüntü indirme *(uzun sürebilir)*
4. Veri birleştirme (PlantNet + iNat → combined_split_v2)
5. tf.data pipeline + augmentation
6. Sınıf ağırlıkları
7. Model mimarisi (EfficientNetB3)
8. Aşama 1: Frozen backbone eğitimi
9. Aşama 2: Fine-tune
10. Değerlendirme + sınıf bazlı accuracy
11. Model + class_names kaydet
12. TFLite dönüştürme
13. TTA testi
"""
))

# ===========================================================================
# HUCRE 1 — Imports + Sabitler
# ===========================================================================
cells.append(code(
"""# === KUTUPHANELER ===
import os, json, time, random, shutil
from pathlib import Path
import numpy as np
import requests
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input

# === YOL TANIMLARI ===
PROJECT_ROOT = Path(".").resolve()
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
NAMES_DIR    = PROJECT_ROOT / "class_names"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
for d in [MODELS_DIR, NAMES_DIR, OUTPUTS_DIR]:
    d.mkdir(exist_ok=True)

# === SABITLER ===
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42
N_INAT      = 300   # Her tur icin iNaturalist'ten indirilecek goruntu
AUTOTUNE    = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
"""
))

# ===========================================================================
# HUCRE 2 — Tur listesi (93 sinif)
# ===========================================================================
cells.append(code(
"""# === 93 TUR TANIMI ===
# PlantNet ID -> Bilimsel isim
# Cikarilan 7 tur: duplikat/cok benzer/cok dusuk performansli
SPECIES = {
    "1355868": "Rosa canina",
    "1355937": "Quercus pubescens",
    "1355978": "Alnus glutinosa",
    "1355990": "Carpinus betulus",
    "1356022": "Populus tremula",
    "1356075": "Fraxinus excelsior",
    "1356126": "Acer pseudoplatanus",
    "1356257": "Tilia platyphyllos",
    "1356382": "Prunus spinosa",
    "1356420": "Malus sylvestris",
    "1356421": "Pyrus communis",
    "1356428": "Sorbus aucuparia",
    "1356692": "Robinia pseudoacacia",
    "1356781": "Gleditsia triacanthos",
    "1356816": "Cercis siliquastrum",
    "1357330": "Pinus sylvestris",
    "1357379": "Pinus pinea",
    "1357677": "Larix decidua",
    "1357705": "Pseudotsuga menziesii",
    "1358094": "Ilex aquifolium",
    "1358133": "Juglans regia",
    "1358150": "Paulownia tomentosa",
    "1358605": "Ficus carica",
    "1358689": "Morus alba",
    "1358751": "Betula pubescens",
    "1358766": "Corylus maxima",
    "1359197": "Galega officinalis",
    "1359483": "Prunus domestica",
    "1359485": "Prunus padus",
    "1359498": "Sorbus torminalis",
    "1359517": "Trifolium pratense",
    "1359525": "Fragaria vesca",
    "1359616": "Cytisus scoparius",
    "1359620": "Ulex europaeus",
    "1359622": "Genista tinctoria",
    "1359669": "Wisteria sinensis",
    "1360153": "Rhamnus cathartica",
    "1360588": "Koelreuteria paniculata",
    "1360590": "Rhododendron ponticum",
    "1360671": "Arbutus unedo",
    "1360811": "Ligustrum lucidum",
    "1360978": "Buddleja davidii",
    "1360998": "Callistemon citrinus",
    "1361024": "Myrtus communis",
    "1361656": "Juniperus horizontalis",
    "1361666": "Thuja occidentalis",
    "1361759": "Magnolia grandiflora",
    "1361823": "Magnolia kobus",
    "1361847": "Liriodendron tulipifera",
    "1361850": "Liquidambar styraciflua",
    "1362294": "Camellia sinensis",
    "1362490": "Philadelphus coronarius",
    "1362927": "Berberis thunbergii",
    "1362954": "Clematis montana",
    "1363021": "Pistacia lentiscus",
    "1363110": "Ceanothus thyrsiflorus",
    "1363126": "Hibiscus syriacus",
    "1363128": "Lavandula angustifolia",
    "1363130": "Rosmarinus officinalis",
    "1363227": "Nerium oleander",
    "1363336": "Jasminum officinale",
    "1363451": "Phyllostachys aurea",
    "1363490": "Agave americana",
    "1363699": "Eucalyptus globulus",
    "1363737": "Trifolium dubium",
    "1363740": "Trifolium repens",
    "1363749": "Caragana arborescens",
    "1363764": "Styphnolobium japonicum",
    "1363778": "Acacia retinodes",
    "1363871": "Barbarea verna",
    "1364099": "Centranthus ruber",
    "1367432": "Lupinus polyphyllus",
    "1369887": "Trachelospermum jasminoides",
    "1369960": "Tradescantia spathacea",
    "1372016": "Morinda citrifolia",
    "1374048": "Tagetes erecta",
    "1385937": "Zamioculcas zamiifolia",
    "1389231": "Phedimus spurius",
    "1389297": "Cereus jamacaru",
    "1389307": "Sedum pachyphyllum",
    "1391112": "Chaerophyllum hirsutum",
    "1391192": "Cirsium eriophorum",
    "1391226": "Cirsium oleraceum",
    "1391483": "Cucurbita maxima",
    "1391652": "Daphne mezereum",
    "1391797": "Dryas octopetala",
    "1391953": "Epipactis atrorubens",
    "1391963": "Epipactis palustris",
    "1392094": "Erucastrum incanum",
    "1392654": "Gomphocarpus physocarpus",
    "1392695": "Hebe salicifolia",
    "1392777": "Hippophae rhamnoides",
    "1393241": "Hypericum calycinum",
}

CLASS_NAMES = sorted(f"plantnet__{pid}" for pid in SPECIES)
NUM_CLASSES = len(CLASS_NAMES)
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
print(f"Toplam sinif: {NUM_CLASSES}")
"""
))

# ===========================================================================
# HUCRE 3 — iNaturalist goruntu indirme
# ===========================================================================
cells.append(code(
"""# === iNATURALIST GORUNTU INDIRME ===
# Her tur icin research-grade, farkli aci/isik/arka plandan 300 goruntu indir
# Daha once indirilenler atlanir (devam ettirilebilir)

INAT_DIR = DATA_DIR / "inat_images"
INAT_DIR.mkdir(exist_ok=True)

def get_taxon_id(scientific_name):
    r = requests.get(
        "https://api.inaturalist.org/v1/taxa",
        params={"q": scientific_name, "per_page": 5},
        headers={"Accept": "application/json"},
        timeout=15, verify=False
    )
    for res in r.json().get("results", []):
        if res.get("name", "").lower() == scientific_name.lower() and res.get("rank") == "species":
            return res["id"]
    results = r.json().get("results", [])
    for res in results:
        if res.get("rank") == "species":
            return res["id"]
    return None

def download_species(scientific_name, n_target=N_INAT):
    folder = INAT_DIR / scientific_name.replace(" ", "_")
    folder.mkdir(exist_ok=True)
    existing = {p.stem for p in folder.glob("*.jpg")}
    if len(existing) >= n_target:
        return len(existing)

    taxon_id = get_taxon_id(scientific_name)
    if not taxon_id:
        print(f"  UYARI: taxon bulunamadi -> {scientific_name}")
        return 0

    downloaded = len(existing)
    for page in range(1, 8):
        if downloaded >= n_target:
            break
        try:
            r = requests.get(
                "https://api.inaturalist.org/v1/observations",
                params={
                    "taxon_id": taxon_id, "quality_grade": "research",
                    "photos": "true", "per_page": 200, "page": page,
                    "order": "random",
                },
                headers={"Accept": "application/json"},
                timeout=30, verify=False
            )
            obs_list = r.json().get("results", [])
        except Exception as e:
            print(f"  Hata (sayfa {page}): {e}")
            break
        if not obs_list:
            break

        for obs in obs_list:
            if downloaded >= n_target:
                break
            photos = obs.get("photos", [])
            if not photos:
                continue
            pid = str(photos[0].get("id", ""))
            if pid in existing:
                continue
            url = photos[0].get("url", "").replace("/square.", "/medium.")
            if not url.startswith("http"):
                continue
            try:
                img_r = requests.get(url, timeout=20, verify=False)
                if img_r.status_code == 200 and len(img_r.content) > 5000:
                    (folder / f"{pid}.jpg").write_bytes(img_r.content)
                    existing.add(pid)
                    downloaded += 1
            except:
                pass
            time.sleep(0.12)

        time.sleep(0.5)

    return downloaded

# Tum turleri indir
print(f"{NUM_CLASSES} tur icin iNaturalist indirme basliyor...")
print("Her tur ~300 goruntu. Kesintiye ugrarsa yeniden calistir, devam eder.\\n")

for i, (pid, sci_name) in enumerate(SPECIES.items(), 1):
    n = download_species(sci_name)
    print(f"[{i:>3}/{NUM_CLASSES}] {sci_name:<40} {n} goruntu")
    time.sleep(0.3)

print("\\nIndirme tamamlandi!")
"""
))

# ===========================================================================
# HUCRE 4 — Veri birlestirme (PlantNet + iNat -> combined_split_v2)
# ===========================================================================
cells.append(code(
"""# === VERI BIRLESTIRME ===
# PlantNet mevcut split + iNaturalist yeni goruntuleri -> combined_split_v2
# iNat bolumu: 240 train / 30 val / 30 test

PLANTNET_SPLIT = DATA_DIR / "combined_split"   # mevcut PlantNet bolumu
OUT_SPLIT      = DATA_DIR / "combined_split_v2"
INAT_TRAIN, INAT_VAL, INAT_TEST = 240, 30, 30

for split in ["train", "val", "test"]:
    (OUT_SPLIT / split).mkdir(parents=True, exist_ok=True)

total_train, total_val, total_test = 0, 0, 0

for cls_name in CLASS_NAMES:
    pid = cls_name.replace("plantnet__", "")
    sci_name = SPECIES[pid]

    # PlantNet goruntuleri kopyala
    for split in ["train", "val", "test"]:
        src = PLANTNET_SPLIT / split / cls_name
        dst = OUT_SPLIT / split / cls_name
        dst.mkdir(exist_ok=True)
        if src.exists():
            for img in src.glob("*.jpg"):
                shutil.copy2(img, dst / img.name)

    # iNat goruntuleri ekle
    inat_dir = INAT_DIR / sci_name.replace(" ", "_")
    if inat_dir.exists():
        imgs = list(inat_dir.glob("*.jpg"))
        random.shuffle(imgs)
        splits_inat = [
            ("train", imgs[:INAT_TRAIN]),
            ("val",   imgs[INAT_TRAIN:INAT_TRAIN + INAT_VAL]),
            ("test",  imgs[INAT_TRAIN + INAT_VAL:INAT_TRAIN + INAT_VAL + INAT_TEST]),
        ]
        for split_name, split_imgs in splits_inat:
            dst = OUT_SPLIT / split_name / cls_name
            dst.mkdir(exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dst / img.name)

# Sayim
for split in ["train", "val", "test"]:
    total = sum(len(list((OUT_SPLIT / split / c).glob("*.jpg"))) for c in CLASS_NAMES if (OUT_SPLIT / split / c).exists())
    print(f"{split:5s}: {total:>6} goruntu ({total // NUM_CLASSES:.0f} ort./sinif)")
"""
))

# ===========================================================================
# HUCRE 5 — tf.data pipeline + augmentation
# ===========================================================================
cells.append(code(
"""# === TF.DATA PIPELINE ===
# Agresif augmentation: her acidan, her isiktan, farkli arka planlardan tanim icin

TRAIN_DIR = str(OUT_SPLIT / "train")
VAL_DIR   = str(OUT_SPLIT / "val")
TEST_DIR  = str(OUT_SPLIT / "test")

def make_dataset(directory, shuffle=False, augment=False):
    ds = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    def preprocess(img, label):
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda img, lbl: (aug_layer(img, training=True), lbl),
                    num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

# Agresif augmentation — farkli aci, isik, zoom, kaydirma
aug_layer = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.30),            # +/-108 derece
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.30),
    layers.RandomBrightness(0.25),
    layers.RandomTranslation(0.10, 0.10),
], name="augmentation")

train_ds = make_dataset(TRAIN_DIR, shuffle=True, augment=True)
val_ds   = make_dataset(VAL_DIR,   shuffle=False, augment=False)
test_ds  = make_dataset(TEST_DIR,  shuffle=False, augment=False)

print("Train:", train_ds.cardinality().numpy(), "batch")
print("Val:  ", val_ds.cardinality().numpy(),   "batch")
print("Test: ", test_ds.cardinality().numpy(),   "batch")
"""
))

# ===========================================================================
# HUCRE 6 — Sinif agirliklari
# ===========================================================================
cells.append(code(
"""# === SINIF AGIRLIKLARI ===
# Siniflar arasi goruntu sayisi farki varsa dengele
from sklearn.utils.class_weight import compute_class_weight

all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(int(labels.numpy()))
all_labels = np.array(all_labels)

weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=all_labels)
class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

print(f"Agirlik araliği: {min(weights):.3f} - {max(weights):.3f}")
"""
))

# ===========================================================================
# HUCRE 7 — Model mimarisi (EfficientNetB3)
# ===========================================================================
cells.append(code(
"""# === MODEL MIMARISI ===
# EfficientNetB3 — ImageNet pretrained backbone
# GlobalAveragePooling -> Dropout(0.4) -> Dense(NUM_CLASSES, softmax)

base_model = keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
)
base_model.trainable = False  # Asama 1'de dondurulmus

inputs  = keras.Input(shape=IMG_SIZE + (3,))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                       kernel_regularizer=keras.regularizers.l2(1e-4))(x)

model = keras.Model(inputs, outputs)
model.summary(line_length=80, show_trainable=True)
"""
))

# ===========================================================================
# HUCRE 8 — Asama 1: Frozen backbone egitimi
# ===========================================================================
cells.append(code(
"""# === ASAMA 1: FROZEN BACKBONE ===
# Sadece ust katmanlar egitilir, backbone donuk
# Amac: yeni siniflar icin iyi bir baslangic noktasi olusturmak

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb_checkpoint = keras.callbacks.ModelCheckpoint(
    str(MODELS_DIR / "efficientnetb3_stage1.keras"),
    save_best_only=True, monitor="val_accuracy", verbose=1,
)
cb_earlystop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1,
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[cb_checkpoint, cb_earlystop],
    class_weight=class_weight_dict,
)

print("\\nAsama 1 tamamlandi.")
print(f"En iyi val_acc: {max(history1.history['val_accuracy']):.4f}")
"""
))

# ===========================================================================
# HUCRE 9 — Asama 2: Fine-tune
# ===========================================================================
cells.append(code(
"""# === ASAMA 2: FINE-TUNE ===
# Son 40 katman acilir, cok kucuk lr ile ince ayar yapilir

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 40
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"Egitilen katman: {trainable_count}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cb_checkpoint2 = keras.callbacks.ModelCheckpoint(
    str(MODELS_DIR / "efficientnetb3_93classes.keras"),
    save_best_only=True, monitor="val_accuracy", verbose=1,
)
cb_earlystop2 = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1,
)
cb_reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[cb_checkpoint2, cb_earlystop2, cb_reduce_lr],
    class_weight=class_weight_dict,
)

print("\\nAsama 2 tamamlandi.")
print(f"En iyi val_acc: {max(history2.history['val_accuracy']):.4f}")
"""
))

# ===========================================================================
# HUCRE 10 — Degerlendirme + sinif bazli accuracy
# ===========================================================================
cells.append(code(
"""# === DEGERLENDIRME ===
import pandas as pd

# Test seti genel accuracy
loss, acc = model.evaluate(test_ds, verbose=0)
print(f"Test Loss    : {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}  ({acc*100:.1f}%)")

# Sinif bazli accuracy
y_true, y_pred = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

per_class = []
for i, cls_name in enumerate(CLASS_NAMES):
    mask = y_true == i
    if mask.sum() == 0:
        continue
    cls_acc = (y_pred[mask] == i).mean() * 100
    pid = cls_name.replace("plantnet__", "")
    sci_name = SPECIES.get(pid, cls_name)
    per_class.append({"Tur": sci_name, "Accuracy": round(cls_acc, 1)})

df = pd.DataFrame(per_class).sort_values("Accuracy")
print(f"\\n%70 alti siniflar: {(df['Accuracy'] < 70).sum()}")
print(f"%80 ve ustu siniflar: {(df['Accuracy'] >= 80).sum()}")
print("\\n--- EN DUSUK 15 ---")
print(df.head(15).to_string(index=False))
print("\\n--- EN YUKSEK 10 ---")
print(df.tail(10).to_string(index=False))
"""
))

# ===========================================================================
# HUCRE 11 — Kaydet (model + class_names)
# ===========================================================================
cells.append(code(
"""# === KAYDET ===
# Model
model_path = MODELS_DIR / "efficientnetb3_93classes.keras"
model.save(str(model_path))
print("Model kaydedildi:", model_path)

# Class names (tur listesi)
names_path = NAMES_DIR / "class_names.json"
with open(names_path, "w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)
print("Class names kaydedildi:", names_path)

# Bilimsel isim haritasi (ID -> isim)
id_map_path = NAMES_DIR / "plantnet_species_id_map.json"
with open(id_map_path, "w", encoding="utf-8") as f:
    json.dump(SPECIES, f, ensure_ascii=False, indent=2)
print("ID haritasi kaydedildi:", id_map_path)

# Egitim gecmisi
hist_all = {}
for k in history1.history:
    hist_all[k] = history1.history[k]
    if k in history2.history:
        hist_all[k] += history2.history[k]
with open(OUTPUTS_DIR / "training_history.json", "w") as f:
    json.dump(hist_all, f, indent=2)
print("Egitim gecmisi kaydedildi.")
"""
))

# ===========================================================================
# HUCRE 12 — TFLite donusturme
# ===========================================================================
cells.append(code(
"""# === TFLITE DONUSTURME ===
# Mobil uygulama icin TFLite modeli olustur

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []   # Quantization yok — tam hassasiyet
tflite_model = converter.convert()

tflite_path = MODELS_DIR / "plant_model_93classes.tflite"
tflite_path.write_bytes(tflite_model)
print(f"TFLite kaydedildi: {tflite_path}")
print(f"Boyut: {tflite_path.stat().st_size / 1024 / 1024:.1f} MB")

# class_names.json'u da models/ klasorune kopyala (uygulama icin)
import shutil
shutil.copy(str(NAMES_DIR / "class_names.json"), str(MODELS_DIR / "class_names.json"))
print("class_names.json modeller klasorune kopyalandi.")
"""
))

# ===========================================================================
# HUCRE 13 — TTA testi
# ===========================================================================
cells.append(code(
"""# === TEST TIME AUGMENTATION (TTA) ===
# Test setinde TTA vs normal tahmin karsilastirmasi
# Model agirliklarini degistirmez

def predict_with_tta(model, img_batch, n_variants=8):
    def prep(arr):
        return preprocess_input(arr.copy().astype("float32"))
    variants = []
    for k in range(4):
        r = np.rot90(img_batch, k=k, axes=(1, 2))
        variants.append(prep(r))
        variants.append(prep(r[:, :, ::-1, :]))
    preds = [model.predict(v, verbose=0) for v in variants[:n_variants]]
    return np.mean(preds, axis=0)

y_true_tta, y_pred_normal, y_pred_tta = [], [], []

for imgs, labels in test_ds:
    imgs_np = imgs.numpy()
    lbls_np = labels.numpy()
    p_normal = model.predict(imgs_np, verbose=0)
    p_tta    = predict_with_tta(model, imgs_np)
    y_true_tta.extend(lbls_np)
    y_pred_normal.extend(np.argmax(p_normal, axis=1))
    y_pred_tta.extend(np.argmax(p_tta,    axis=1))

y_true_tta    = np.array(y_true_tta)
y_pred_normal = np.array(y_pred_normal)
y_pred_tta    = np.array(y_pred_tta)

acc_normal = (y_pred_normal == y_true_tta).mean()
acc_tta    = (y_pred_tta    == y_true_tta).mean()

print(f"Normal Accuracy : {acc_normal:.4f}  ({acc_normal*100:.1f}%)")
print(f"TTA    Accuracy : {acc_tta:.4f}  ({acc_tta*100:.1f}%)")
print(f"TTA kazanimi    : +{(acc_tta - acc_normal)*100:.2f} puan")
"""
))

# ===========================================================================
# Notebook JSON olustur ve kaydet
# ===========================================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook olusturuldu: {OUT}")
print(f"Toplam hucre: {len(cells)}")
