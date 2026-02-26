# Univariate-Time-Series-Forecasting
The task is to explore machine learning techniques and develop a data-driven model for hourly forecasts of  future energy demand accurately.

# Electricity Consumption Forecasting a complete ML Pipeline

A comprehensive end-to-end Machine Learning pipeline for forecasting **hourly electricity consumption (MWh)** using statistical, machine learning, and deep learning models.

---

## Project Overview

This project builds and benchmarks multiple forecasting models to predict hourly electricity demand using historical time-series data.

The pipeline includes:

- Data preprocessing
- Feature engineering
- Baseline modeling
- Statistical models (ARIMA)
- Additive models (Prophet)
- Machine learning models (XGBoost, LightGBM)
- Deep learning model (LSTM)
- Model comparison & evaluation

---

## Dataset Description

The dataset contains:

- `Start time UTC`
- `End time UTC`
- `Electricity consumption (MWh)`

After preprocessing:

- timestamp
- consumption

Data Characteristics:

- Hourly frequency
- Univariate time series
- Daily and weekly seasonality
- Stationary

---

## Machine Learning Pipeline

### Phase 1: Data Loading & Inspection
- Parse timestamps
- Sort chronologically
- Check missing values and duplicates

### Phase 2: Data Preprocessing
- Handle missing values (forward fill / interpolation)
- Remove outliers using IQR method

### Phase 3: Exploratory Data Analysis
- Seasonal decomposition (Trend + Seasonality + Residual)
- Stationarity check (ADF test)
- ACF & PACF analysis

### Phase 4: Feature Engineering
- Lag features (1, 3, 6, 12, 24, 48, 168 hours)
- Rolling statistics (mean, std, min, max)
- Temporal features (hour, day, month)
- Cyclical encoding (sin/cos transformations)

### Phase 5: Train / Validation / Test Split
- 70% Train
- 15% Validation
- 15% Test
- No random shuffling (time-series safe)

### Phase 6–10: Model Training
- Naive Baseline
- Seasonal Naive
- ARIMA
- Prophet
- LSTM
- XGBoost
- LightGBM

### Phase 11: Model Comparison
- Rank models by MAE
- Visualize performance metrics

### Phase 12: Reporting
- Generate final performance summary
- Save all plots automatically

---

## Models Implemented

### Baseline Models
- Naive Mean Forecast
- Seasonal Naive (24-hour lag)

### Statistical Model
- ARIMA (AutoRegressive Integrated Moving Average)

### Additive Model
- Prophet

### Deep Learning
- LSTM (Long Short-Term Memory)

### Tree-Based Gradient Boosting
- XGBoost
- LightGBM

---

## Evaluation Metrics

The following metrics are used for comparison:

### MAE (Mean Absolute Error)
Average absolute prediction error in MWh.

### RMSE (Root Mean Squared Error)
Penalizes large prediction errors.

### MAPE (Mean Absolute Percentage Error)
Percentage-based interpretability.

Lower values indicate better performance.

---

## Generated Visualizations

The pipeline automatically generates:

- `01_raw_data_analysis.png`
- `02_seasonal_decomposition.png`
- `03_acf_pacf.png`
- `04_feature_analysis.png`
- `05_prophet_forecast.png`
- `06_lstm_training_history.png`
- `07_xgboost_feature_importance.png`
- `08_model_comparison.png`
- `09_predictions_visualization.png`

---

