# Air Pollution Prediction Index

This project focuses on predicting the Air Quality Index (AQI) using multiple machine learning and deep learning models, based on air pollutant data collected from Indian cities. The complete pipeline covers data cleaning, feature engineering, model training, evaluation, and deployment using Streamlit.

---

## Introduction

This project aims to develop machine learning models to predict the Air Quality Index (AQI) based on historical air quality data. The models are trained on the [Air Quality Data in India (2015 - 2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) dataset and evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).


## 📁 Dataset

- **Source**: [Air Quality Data in India (2015 - 2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Features**:  
  `Date`, `City`, `PM2.5`, `PM10`, `NO`, `NO₂`, `NOx`, `NH₃`, `CO`, `SO₂`, `O₃`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI Label`

---

## 🧠 Models Overview

### ✅ **Model 1: Regression-Based AQI Prediction (Multi-feature Approach)**

**Objective**: Predict AQI using multiple pollutants via ensemble and linear models.

- **Cleaning & Preprocessing**:
  - Removed irrelevant columns: `NOx`, `NH₃`, `Benzene`, `Toluene`, `Xylene`
  - Imputed missing values via **three-step interpolation** grouped by `City`, `AQI Label`, and `Date`
  - Detected and replaced outliers with AQI-label-wise mean values
- **Feature Engineering**:
  - Selected 6 pollutants: `PM2.5`, `PM10`, `NO₂`, `CO`, `SO₂`, `O₃`
  - Computed **weekly rolling averages** (e.g., `PM2.5_rw_avg`)
  - Scaled features and split data (80% train / 20% test)
- **Models Tried**:
  - SGDRegressor, XGBoost, RandomForest, LGBM
- **Hyperparameter Tuning**:
  - Used RandomizedSearchCV with cross-validation
- **Final Models Benchmarked**:

| Model                 | R² Score | MSE      |
|----------------------|----------|----------|
| XGBoost              | 0.9327   | 780.43   |
| GradientBoosting     | 0.8349   | 1913.66  |
| **RandomForest** ✅   | **0.9490** | **591.35** |

📌 **Winner**: RandomForestRegressor (best balance of accuracy and efficiency)

---

### ✅ **Model 2: Lightweight AQI Prediction (PM2.5 & PM10 Focus)**

**Objective**: Build a simple yet effective model using just the two most critical features for AQI – PM2.5 and PM10.

- **Preprocessing**:
  - Missing values filled using AQI-label-wise interpolation
  - Enhanced outlier detection using **EllipticEnvelope** (contamination = 0.30)
- **Feature Selection**:
  - Only `PM2.5` and `PM10`
  - Dataset after cleaning: **52,000 instances** (70% train / 30% test)
- **Models Tried**:
  - ElasticNet, SGD, KNeighbors, ExtraTrees, Lasso, Lars, BayesianRidge, OMP
- **Final Model Evaluation**:

| Model         | MSE      |
|---------------|----------|
| ElasticNet    | 516.61   |
| KNeighbors    | **367.92** |
| **SGDRegressor** ✅ | 516.64   |

📌 **Winner**: **SGDRegressor** – selected based on overall performance and consistency through visual inspection

---

### ✅ **Model 3: Neural Network-Based AQI Prediction**

**Objective**: Leverage deep learning to capture non-linear relationships among pollutants.

- **Features Used**:  
  `PM2.5`, `PM10`, `NO₂`, `CO`, `SO₂`, `O₃`
- **Neural Network Architectures**:

| Model       | Layers                        | Params | Patience | Iterations | RMSE     |
|-------------|-------------------------------|--------|----------|------------|----------|
| Model_91    | [6 → 6 → 1]                   | 91     | 2        | 8          | 26.1955  |
| Model_169   | [12 → 6 → 1]                  | 169    | 2        | 24         | 25.8530  |
| **Model_187** ✅ | [12 → 6 → 3 → 1]             | 187    | 3        | 53         | **24.9424** |

- **Training**:
  - Loss: MSE, Optimizer: Adam (lr = 0.01), Metrics: RMSE & MSE
  - Early stopping used to avoid overfitting (patience: 2–3)

📌 **Winner**: **Model_187** – deeper architecture yielded best performance

---

## 🚀 Deployment

All selected models were deployed using **Streamlit** to enable real-time AQI prediction:

- **Deployed Models**:
  - `RandomForestRegressor` (Model 1)
  - `SGDRegressor` (Model 2)
  - `Model_91`, `Model_169`, and `Model_187` (Model 3)
- **Features**:
  - User inputs pollutant values for the respective model
  - Model returns **predicted AQI** dynamically
  - Clean UI with sliders, number fields, and model-specific prediction panels

## Results

<img width="1138" alt="Screenshot 2025-06-27 at 2 16 37 PM" src="https://github.com/user-attachments/assets/c708ab25-470d-46c9-b1d8-6870b5cad438" />
<img width="1138" alt="Screenshot 2025-06-27 at 2 20 20 PM" src="https://github.com/user-attachments/assets/e7a49b1f-70c3-4cce-86bd-da4ffce5a523" />
<img width="966" alt="Screenshot 2025-06-27 at 2 20 38 PM" src="https://github.com/user-attachments/assets/0df46992-60e3-4d15-ba45-66efd472f231" />
<img width="966" alt="Screenshot 2025-06-27 at 2 20 44 PM" src="https://github.com/user-attachments/assets/83495e86-928b-4868-94db-d4e5413abcf3" />
<img width="1170" alt="Screenshot 2025-06-27 at 2 20 55 PM" src="https://github.com/user-attachments/assets/9b8072e4-ba9f-4928-8df0-949d735ed2af" />
<img width="966" alt="Screenshot 2025-06-27 at 2 21 14 PM" src="https://github.com/user-attachments/assets/d14f0d30-3e9e-4039-a642-ca55c3100268" />
<img width="1138" alt="Screenshot 2025-06-27 at 2 16 37 PM" src="https://github.com/user-attachments/assets/6d5d1cf4-2ba5-4c1c-a044-d9e465a2b8b4" />


---

## 📌 Final Takeaways

- Multiple modeling strategies were explored—traditional ML and deep learning—to predict AQI with varying complexity and accuracy.
- Outlier handling, thoughtful imputation, and context-aware feature engineering played a crucial role in boosting model performance.
- Streamlit deployment provides an accessible interface for end-users to interact and visualize model predictions in real time.
