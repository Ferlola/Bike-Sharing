https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

# 🚲 Bike Sharing Demand Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![R2 Score](https://img.shields.io/badge/R²-0.99-brightgreen.svg)

---

## 📌 Project Overview

This project tackles a **supervised regression problem** to predict hourly bike rental demand using the **Bike Sharing Dataset (UCI)**.

The goal is to model the target variable `cnt` (total rentals) leveraging:

* Temporal patterns
* Weather conditions
* Advanced feature engineering

The final model achieves **near-perfect performance (R² ≈ 0.99)**, highlighting strong predictive power when combining cyclical encoding and gradient boosting.

---

## 📊 Dataset

* Source: UCI Machine Learning Repository
* Granularity: **Hourly observations**
* Problem type: Regression

### Key Features:

* **Temporal**: date, hour, weekday, season
* **Weather**: temperature, humidity, windspeed, weather situation
* **Target**: `cnt` (total bike rentals)

---

## 🧠 Feature Engineering

A key strength of this project lies in **domain-driven feature engineering**:

### 🔁 Cyclical Encoding

Captures periodic patterns:

* Month, weekday, hour, day of year → `sin/cos`

### 📅 Temporal Features

* Year, quarter
* Weekend indicator
* Trend (normalized time progression)

### ⚙️ Derived Features

* `temp_hum` (interaction)
* `temp_atemp`
* `rush_hour` (peak commuting hours)
* `clima_season` interaction

These transformations allow the model to learn **non-linear seasonal behavior** effectively.

---

## 🤖 Model

### HistGradientBoostingRegressor (scikit-learn)

Chosen for:

* High performance on tabular data
* Native handling of non-linearities
* Efficient training via histogram binning
* Built-in early stopping

---

## 🔍 Hyperparameter Optimization

Optimization performed using **Optuna**:

* **Sampler**: TPE (Tree-structured Parzen Estimator)
* **Pruner**: Median Pruner (early stopping of bad trials)
* **Validation**: 5-Fold Cross Validation


Additional strategies:

* Warm-start initialization
* Early stopping inside model training

---

## 📈 Model Performance

### 🎯 Test Set Metrics

| Metric | Value      |
| ------ | ---------- |
| RMSE   | 3.7948     |
| MAE    | 1.7043     |
| R²     | **0.9995** |
| MAPE   | 1.40%      |
| Bias   | -0.0109    |

### ✅ Key Insights:

* Excellent generalization
* Very low residual bias
* Stable performance across demand segments

---

## 📊 Exploratory & Model Analysis

A complete analytical dashboard was built including:

### 📉 Error Analysis

* Residual distribution
* Residuals vs predictions
* CDF of absolute error

### 📊 Segment Evaluation

* RMSE by demand range
* Bias by segment
* Outlier detection

### 🌦️ Feature Insights

* Demand vs temperature
* Demand vs weather conditions
* Demand vs humidity & wind

### 🔥 Behavioral Patterns

* Heatmap: **hour vs weekday demand**
* Peak-hour detection
* Seasonal usage trends

---

## 🧪 Validation Strategy

* Train/Test split: **80/20**
* Cross-validation: **KFold (k=5)**
* Robust evaluation on unseen data

> ⚠️ Note: A time-based split (TimeSeriesSplit) could further improve real-world validation.

---


## 🛠️ Tech Stack

* **Python**
* **pandas / numpy**
* **scikit-learn**
* **Optuna**
* **matplotlib / seaborn**

---

## 🚀 Key Takeaways

* Feature engineering is **critical** in tabular ML problems
* Gradient boosting models excel with structured + temporal data
* Cyclical encoding significantly improves performance
* Proper validation and tuning can push models close to optimal


