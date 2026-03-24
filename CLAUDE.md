# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install dependencies into the virtual environment:
```bash
pip install -r requirements.txt
```

Download datasets from Kaggle (required before running training):
```bash
kaggle datasets download -d xhlulu/leafsnap-dataset
kaggle datasets download -d noahbadoa/plantnet-300k-images
```
Extract into `data/leafsnap-dataset/` and `data/plantnet_300K/` respectively.

Validate the save/load infrastructure before training:
```bash
python src/save_load_smoketest.py
```

Run the full training pipeline:
```bash
python src/main.py
```

Launch Jupyter for interactive work:
```bash
jupyter notebook notebooks/egitilmis_model.ipynb
```

## Architecture

This is a **multi-dataset plant species classification** system using transfer learning on 250 species (50 from Leafsnap + 200 from PlantNet-300K).

### Model

MobileNetV2 (ImageNet pretrained) backbone → GlobalAveragePooling2D → Dropout(0.4) → Dense(250, softmax, L2 regularized). Input: 224×224×3.

**Two-stage training:**
1. Backbone frozen — train top layers only (Adam lr=1e-3, 8 epochs)
2. Last 35 backbone layers unfrozen — fine-tune (Adam lr=3e-6, 12 epochs with early stopping + checkpointing, class-weighted loss)

### Data Pipeline

- `src/paths.py` — central `pathlib.Path` definitions for all project directories
- `src/main.py` — full end-to-end pipeline: dataset download → preprocessing → subset selection → combined dataset creation → training → evaluation
- `notebooks/egitilmis_model.ipynb` — interactive version of the same pipeline with outputs

**Dataset construction:**
- Leafsnap: top 50 species with ≥30 images, capped at 80/class → 70/15/15 train/val/test split
- PlantNet: 200 random classes with ≥50 images, limited to 80 train / 20 val / 20 test per class
- Combined class names use prefixes: `leafsnap__<species>` and `plantnet__<species>`
- Final split stored in `data/combined_split/`, class names in `class_names/class_names.json`

**Data augmentation** (applied during training): horizontal flip, ±10° rotation, ±15% zoom, ±10% contrast.

### Saved Artifacts

All saved artifacts are gitignored:
- `models/best_finetuned_model_v2.keras` — latest model
- `class_names/class_names.json` — 250-element JSON array of class names (must be loaded alongside the model for inference)
- `outputs/history_plant_model_250classes_local.json` — training history
