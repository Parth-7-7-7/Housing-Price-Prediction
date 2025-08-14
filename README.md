# 🏠 Real Estate Price Prediction

## 📌 Project Overview
This project predicts **housing prices** based on various features such as location, property attributes, and socio-economic indicators.  
We use the **Boston Housing-style dataset** (`data.csv`) and apply machine learning regression models to estimate property prices.  

The model uses **data preprocessing pipelines**, **feature engineering**, and **Random Forest Regression** for accurate predictions.

---

## 🗂 Dataset
- **Source:** `data.csv` (provided in the project)
- **Description:** Includes housing-related attributes such as:
  - **CRIM** – Crime rate
  - **ZN** – Residential land zoning proportion
  - **CHAS** – Charles River proximity (categorical)
  - **RM** – Average rooms per dwelling
  - **LSTAT** – % lower status population
  - and other relevant housing features.
- **Target Variable:** Median house value

---

## 🚀 Features
- **Stratified Sampling**: Ensures important categorical features (like `CHAS`) are proportionally split into training and testing datasets.
- **Preprocessing Pipeline**:
  - Handle missing values with **median imputation**.
  - Standardize numerical features using **StandardScaler**.
- **Model Training**:
  - Random Forest Regressor (final model)
  - (Optionally) Linear Regression for comparison.
- **Model Evaluation**:
  - RMSE (Root Mean Squared Error)
  - Cross-validation scores

---

## 🛠 Technologies Used
- **Python 3**
- **Pandas, NumPy** – Data analysis
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Preprocessing, model training, evaluation
- **Joblib** – Model saving (optional for deployment)

---

## 📊 Model Workflow
1. **Data Loading**
   - Load dataset from `data.csv`.
2. **Exploratory Data Analysis (EDA)**
   - Summary statistics and visualizations.
3. **Data Splitting**
   - Use `StratifiedShuffleSplit` to split data ensuring balanced `CHAS` feature.
4. **Preprocessing**
   - Apply imputation for missing values.
   - Standardize numerical features.
5. **Model Training**
   - Train a **Random Forest Regressor**.
6. **Model Evaluation**
   - Evaluate RMSE on test set.
   - Perform **10-fold Cross-Validation** for robustness.
7. **(Optional) Model Saving**
   ```python
   import joblib
   joblib.dump(model, 'real_estate_model.joblib')
