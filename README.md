# 🩸 Blood Cell Anomaly Detection

> A complete tabular ML pipeline for detecting blood cell anomalies — 3 tasks, 4 models, SHAP explainability, and 8 production-grade enhancements.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project builds a full machine learning pipeline on the **CytoDiffusion 2025** dataset — 5,880 blood cell records × 36 morphological and clinical features × 19 cell types. Instead of using raw microscopy images, we work entirely with **tabular features** to see how close we can get to the image-based SOTA.

**Inspired by:** CytoDiffusion — *Nature Machine Intelligence* (2025) · Cambridge · UCL · QMUL  
**SOTA Benchmark:** AUC = 0.990 (image-based vision model)

---

## 🎯 Tasks

| Task | Problem | Model | Metric |
|------|---------|-------|--------|
| 1 | Binary: Normal vs Anomaly | XGBoost + SMOTE | ROC-AUC |
| 2 | Multi-class: 19 cell types | LightGBM + PyTorch MLP | Macro-F1 |
| 3 | Disease-level prediction | XGBoost | Recall |
| 4 | Explainability | SHAP | Feature importance |

---

## ⚠️ Critical Note — Data Leakage

The dataset contains 3 AI-generated columns that directly encode the target label:

```
cytodiffusion_anomaly_score
classification_confidence
labeller_confidence
```

**Without removing these → AUC = 1.000 (not real learning)**  
**After removing these → AUC ≈ 0.88–0.93 (actual generalization)**

This pipeline removes them before any training step.

---

## 🗂️ Project Structure

```
blood-cell-anomaly-detection/
│
├── blood-cell-anomaly-detection-full-pipeline.ipynb   # Main notebook
├── requirements.txt                                    # Dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/goktani/blood-cell-anomaly-detection.git
cd blood-cell-anomaly-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Dataset: [Blood Cell Anomaly Detection 2025](https://www.kaggle.com/datasets/alitaqishah/blood-cell-anomaly-detection-2025) on Kaggle.

Place the CSV files under:
```
/kaggle/input/datasets/alitaqishah/blood-cell-anomaly-detection-2025/
```

Or update the `BASE` path in the notebook to your local directory.

### 4. Run the notebook
```bash
jupyter notebook blood-cell-anomaly-detection-full-pipeline.ipynb
```

---

## 🔬 Pipeline Steps

### Preprocessing
- Label encoding for categorical features
- 3 target vectors: binary, 19-class, disease-level
- Stratified 70/15/15 train/val/test split
- `StandardScaler` fit on train only
- SMOTE oversampling for binary task

### Disease Mapping
| Cell Types | Disease Label |
|-----------|--------------|
| Blast Cell, Prolymphocyte | Leukemia |
| Elliptocyte, Schistocyte, Spherocyte, Target Cell | Anemia |
| Sickle Cell | Sickle Cell Disease |
| Hypersegmented Neutrophil, Toxic Granulation, Reactive Lymphocyte | Infection |
| Smudge Cell, Artefact | Artefact |
| 7 normal types | Normal |

---

## 🧠 Models

### XGBoost (Task 1 & 3)
```python
xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

### LightGBM (Task 2)
```python
lgb.LGBMClassifier(
    n_estimators=400,
    num_leaves=63,
    learning_rate=0.05,
    class_weight='balanced',
)
```

### PyTorch MLP (Task 2)
```
Input(N) → Linear(256) → BN → ReLU → Dropout(0.3)
         → Linear(128) → BN → ReLU → Dropout(0.2)
         → Linear(64)  → ReLU
         → Linear(19)
```
- Loss: `CrossEntropyLoss` with class weights
- Optimizer: `AdamW` lr=1e-3, weight_decay=1e-4
- Scheduler: `CosineAnnealingLR` over 50 epochs
- Checkpoint: best validation Macro-F1

---

## 🔧 Enhancements

| # | Enhancement | Benefit |
|---|-------------|---------|
| 1 | Feature Correlation Removal (|r| > 0.95) | Reduces noise |
| 2 | Stratified K-Fold Cross Validation (k=5) | Reliable estimates |
| 3 | Optuna Hyperparameter Tuning (50 trials) | +AUC improvement |
| 4 | Soft Voting Ensemble (XGB + LGB) | Reduces variance |
| 5 | Hierarchical Classifier (binary → multi-class) | Better rare class recall |
| 6 | Error Analysis | Identifies confusion patterns |
| 7 | Calibration Curve (Platt Scaling) | Trustworthy probabilities |
| 8 | Learning Curve | Data efficiency analysis |

---

## 📊 Results

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Binary | XGBoost | ROC-AUC | ~0.90+ |
| Multi-class | LightGBM | Macro-F1 | ~0.87+ |
| Multi-class | PyTorch MLP | Macro-F1 | ~0.85+ |
| Disease-level | XGBoost | Macro-F1 | ~0.88+ |
| **SOTA** | **CytoDiffusion (images)** | **AUC** | **0.990** |
| Baseline | Paper tabular | AUC | 0.916 |

> Exact scores depend on your environment and random seed.

---

## 💡 Key Takeaways

1. **Data leakage is the #1 pitfall** — always audit features before training
2. **Tabular models approach but don't match image SOTA** — the gap quantifies the information in raw pixels
3. **Recall > Accuracy** for clinical tasks — a missed Leukemia cell is far more costly than a false alarm
4. **SHAP explainability is essential** for medical AI — models must be interpretable to clinicians
5. **Enhancements compound** — each adds 1–3%, combined they matter significantly

---

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
torch>=2.0.0
shap>=0.43.0
optuna>=3.3.0
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- Dataset: [CytoDiffusion 2025 on Kaggle](https://www.kaggle.com/datasets/alitaqishah/blood-cell-anomaly-detection-2025)
- Original paper: *CytoDiffusion*, Nature Machine Intelligence, 2025
- Institutions: Cambridge · UCL · QMUL
