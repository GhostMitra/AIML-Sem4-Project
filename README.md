# 🌾🏥 Machine Learning Projects
## Crop Recommendation System · Liver Disease Prediction System

> Two end-to-end machine learning projects with data preprocessing, model training,
> evaluation, and **Explainable AI (SHAP + LIME)** — built for Jupyter Lab.

---

## 📁 Repository Structure

```
├── Project 1 — Crop Recommendation/
│   ├── Crop_Recommendation_XAI.ipynb   ← Main Jupyter notebook
│   ├── Train_Dataset.csv               ← Training data  (18,079 samples)
│   └── Test_Dataset.csv                ← Test data      (18,079 samples)
│
├── Project 2 — Liver Disease Prediction/
│   ├── LiverDisease_Updated.ipynb      ← Main Jupyter notebook
│   └── Dataset.csv                     ← ILPD dataset   (583 patients)
│
├── requirements.txt                    ← All dependencies
└── README.md                           ← This file
```

---

## 🌾 Project 1 — Intelligent Crop Recommendation System

### Overview
A machine learning system that recommends the most suitable crop for cultivation
based on **soil nutrient levels** and **climatic conditions**.
Farmers input NPK values, soil pH, rainfall, and temperature — the system returns
the optimal crop with a confidence score and a full XAI explanation of *why*.

### Dataset
| Property | Detail |
|---|---|
| **Source** | [IEEE DataPort – Crop Recommendation Dataset](https://ieee-dataport.org/documents/crop-recommendation-dataset) |
| **Size** | 36,158 samples (18,079 train / 18,079 test) |
| **Classes** | 40 crop types |
| **Features** | N, P, K (mg/kg), pH, Rainfall (mm), Temperature (°C) |
| **Missing values** | None |

### Features
| Column | Description | Unit |
|---|---|---|
| `N` | Nitrogen content in soil | mg/kg |
| `P` | Phosphorus content in soil | mg/kg |
| `K` | Potassium content in soil | mg/kg |
| `pH` | Soil acidity/alkalinity | — |
| `rainfall` | Annual rainfall | mm |
| `temperature` | Average temperature | °C |
| `Crop` | **Target** — recommended crop | 40 classes |

### Models Trained
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| ✅ **Decision Tree** *(selected)* | **95.50%** | **95.56%** | **95.50%** | **95.52%** |
| Random Forest | 95.50% | 95.43% | 95.50% | 95.45% |
| KNN | 94.91% | 94.88% | 94.91% | 94.89% |
| SVM | 93.00% | 88.52% | 93.00% | 90.27% |
| Naive Bayes | 92.85% | 88.37% | 92.85% | 90.12% |

> **Best model: Decision Tree** — selected for highest precision (95.56%), full
> interpretability, and fast inference suitable for mobile/edge deployment.

### Notebook Structure (`Crop_Recommendation_XAI.ipynb`)

| Section | Description |
|---|---|
| 1 | Imports & configuration |
| 2 | Load dataset |
| 3 | Exploratory Data Analysis (distributions, boxplots, heatmap) |
| 4 | Data preprocessing — label encoding, StandardScaler |
| 5 | Train 5 ML models |
| 6 | Evaluate — table, radar chart, confusion matrix, feature importance |
| 7 | **SHAP** — global bar, beeswarm, force plots, dependence, per-crop heatmap |
| 8 | **LIME** — individual explanations, 9-sample grid, global importance |
| 9 | SHAP vs LIME comparison |
| 10 | Summary & conclusions |
| 11 | Live crop predictor with LIME explanation |

### XAI Findings
| Method | Top Feature | Key Insight |
|---|---|---|
| **SHAP** | Potassium (K) | K, N, and pH are the primary global drivers |
| **LIME** | pH / Potassium | pH has strong local influence near decision boundaries |
| **Both agree** | Soil nutrients lead | N, P, K collectively outrank climatic features |

---

## 🏥 Project 2 — Liver Disease Prediction System

### Overview
A binary classification system that predicts whether a patient has **liver disease**
using clinical blood test markers from the Indian Liver Patient Dataset (ILPD).
Covers full preprocessing (handling class imbalance), model comparison,
cross-validation, hyperparameter tuning, and XAI explanations.

### Dataset
| Property | Detail |
|---|---|
| **Source** | [UCI ML Repository – ILPD](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset) |
| **Size** | 583 patients |
| **Classes** | 2 (Liver Disease / Healthy) — imbalanced (416 / 167) |
| **Features** | 10 clinical attributes |
| **Missing values** | 4 (in AG Ratio — handled with median imputation) |

### Features
| Column | Description | Unit |
|---|---|---|
| `Age` | Patient age | years |
| `Gender` | Patient gender | Male / Female |
| `TB` | Total Bilirubin | mg/dL |
| `DB` | Direct Bilirubin | mg/dL |
| `Alkphos` | Alkaline Phosphotase | IU/L |
| `Sgpt` | Alamine Aminotransferase | IU/L |
| `Sgot` | Aspartate Aminotransferase | IU/L |
| `TP` | Total Proteins | g/dL |
| `ALB` | Albumin | g/dL |
| `A/G Ratio` | Albumin & Globulin Ratio | — |
| `Selector` | **Target** — 1 = Liver Disease, 2 = Healthy | binary |

### Models Trained
| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 🥇 | **Random Forest** *(selected)* | **86.83%** | **90.67%** | **81.93%** | **86.08%** |
| 🥈 | Decision Tree | 85.63% | 90.41% | 79.52% | 84.62% |
| 🥉 | XGB (GradientBoost) | 82.63% | 87.50% | 75.90% | 81.29% |
| 4 | Logistic Regression | 73.05% | 76.39% | 66.27% | 70.97% |
| 5 | SVM | 74.85% | 88.68% | 56.63% | 69.12% |
| 6 | KNN | 70.66% | 79.31% | 55.42% | 65.25% |

> **Best model: Random Forest** — highest F1 (86.08%) and best balance of
> precision and recall, robust to the class imbalance after upsampling.

### Notebook Structure (`LiverDisease_Updated.ipynb`)

| Section | Description |
|---|---|
| 1 | Imports & configuration |
| 2 | Load dataset |
| 3 | EDA — statistical summary, distributions, class/gender charts, heatmap |
| 4 | Preprocessing — median imputation, label encoding, StandardScaler, upsampling |
| 5 | Feature selection via Random Forest importance |
| 6 | Train 6 ML models |
| 7 | Evaluate — table, bar chart, confusion matrices (2×3 grid), ROC curves |
| 8 | Cross-validation (5-fold stratified) |
| 9 | Hyperparameter tuning (RandomizedSearchCV) |
| 10 | Preprocessing impact comparison |
| 11 | Patient risk predictor function |

---

## 🔍 Explainable AI — SHAP & LIME

Both projects implement SHAP and LIME **from scratch** using only `scikit-learn`
and `numpy`, so no external `shap` or `lime` packages are required to run the
notebooks. The optional packages in `requirements.txt` unlock the official
library versions for extended use.

### SHAP — SHapley Additive exPlanations
```
Theory  : Cooperative game theory — each feature gets its fair share
          of the prediction as the average marginal contribution across
          all possible feature orderings.

Method  : Marginal conditional expectation
          SHAP_j(x) = |f(x) − E[f(x | feature j marginalised)]|
          averaged over background training samples.

Plots   : Global bar · Beeswarm · Force plots · Dependence · Heatmap
```

### LIME — Local Interpretable Model-Agnostic Explanations
```
Theory  : Any complex model can be approximated linearly in a small
          local neighbourhood around a single prediction.

Algorithm:
  1. Perturb — generate noise samples around the input
  2. Query  — get black-box predictions on perturbations
  3. Weight — rank by proximity (Gaussian kernel)
  4. Fit    — weighted Ridge regression (local surrogate)
  5. Read   — surrogate coefficients = feature attributions

Plots   : Individual explanations · 9-sample grid · Global importance
```

### SHAP vs LIME — Key Differences

| Property | SHAP | LIME |
|---|---|---|
| **Scope** | Global + Local | Local only |
| **Theory** | Game theory (Shapley values) | Local linear approximation |
| **Consistency** | Guaranteed by axioms | Depends on kernel & sampling |
| **Speed** | Slower (all coalitions) | Faster (one neighbourhood) |
| **Best for** | Understanding model globally | Explaining one prediction |
| **Output** | Signed feature contributions | Linear surrogate coefficients |

---

## 🚀 Getting Started

### 1. Clone / download the project
```bash
git clone https://github.com/your-username/ml-projects.git
cd ml-projects
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Lab
```bash
jupyter lab
```

### 5. Open a notebook
- **Crop Recommendation** → `Project 1 — Crop Recommendation/Crop_Recommendation_XAI.ipynb`
- **Liver Disease**       → `Project 2 — Liver Disease Prediction/LiverDisease_Updated.ipynb`

> Make sure the CSV data files are in the **same folder** as the notebook before running.

### 6. Run all cells
`Kernel → Restart Kernel and Run All Cells`

---

## 💡 Usage — Live Predictors

### Crop Recommendation
```python
# Predict the best crop for given soil & climate conditions
recommend_crop(
    N=80, P=40, K=40,
    pH=5.66,
    rainfall=297.66,
    temperature=29.57
)
# Output:
# #1  rice                ██████████████████████████    98.7%
# #2  maize               ░░░░░░░░░░░░░░░░░░░░░░░░░░     0.8%
# #3  wheat               ░░░░░░░░░░░░░░░░░░░░░░░░░░     0.5%
```

### Liver Disease Risk
```python
# Predict liver disease risk for a new patient
predict_liver_disease(
    age=45, gender='Male',
    tb=2.1, db=0.8,
    alkphos=230, sgpt=85, sgot=72,
    tp=6.5, alb=3.1, ag_ratio=0.85
)
# Output:
# Prediction  : 🔴 LIVER DISEASE DETECTED
# Probability : 73.50%
```

---

## 📊 Results Summary

### Project 1 — Crop Recommendation
- **Best Model:** Decision Tree — **95.50% accuracy** across 40 crop classes
- **Top SHAP feature:** Potassium (K) followed by Nitrogen (N)
- **Dataset size:** 36,158 samples (perfectly balanced, 40 classes)

### Project 2 — Liver Disease
- **Best Model:** Random Forest — **86.83% accuracy** (binary classification)
- **Top SHAP feature:** Direct Bilirubin and Alkaline Phosphotase
- **Class imbalance handled:** 416 disease / 167 healthy → upsampled to 416/416

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn |
| Boosting | XGBoost / GradientBoostingClassifier |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| XAI | SHAP & LIME (custom implementations + optional packages) |
| Imbalance | imbalanced-learn (upsampling / SMOTE) |
| Environment | Jupyter Lab |

---

## 👥 Authors & Acknowledgements

- Dataset 1: [IEEE DataPort](https://ieee-dataport.org/documents/crop-recommendation-dataset)
- Dataset 2: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)
- SHAP paper: Lundberg & Lee, *NeurIPS 2017* — "A Unified Approach to Interpreting Model Predictions"
- LIME paper: Ribeiro et al., *KDD 2016* — "Why Should I Trust You?"

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
