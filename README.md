# Univariate-Time-Series-Forecasting
The task is to explore machine learning techniques and develop a data-driven model for hourly forecasts of  future energy demand accurately.

## Overview

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

## Dataset Description

The dataset contains 52966 data points or values of energy prediction across time intervals. Mostly the time-intervals are or 1 hours, however fluctuations exists.

Columns:
- `Start time UTC`
- `End time UTC`
- `Electricity consumption (MWh)`

After preprocessing the data attributes are as follows:

- timestamp
- consumption

Data Characteristics:

- Hourly frequency
- Univariate time series
- Daily and weekly seasonality
- Stationary

## Machine Learning Pipeline

### Phase 1: Data Loading & Inspection
- Parse timestamps
- Sort chronologically
- Check missing values and duplicates

### Phase 2: Data Preprocessing
- Handle missing values (forward fill / interpolation)
- Remove outliers using IQR method
- Applying differencing if needed

### Phase 3: Exploratory Data Analysis
- Seasonal decomposition (Trend + Seasonality + Residual)
- Stationarity check (ADF test)
- Autocorrelation & partial autocorrelation (ACF, PACF) plot
- Visualize time-series trends
- Identify seasonality patterns (daily, weekly, monthly)

### Phase 4: Feature Engineering
- Lag features (1, 3, 6, 12, 24, 48, 168 hours)
- Rolling statistics (mean, std, min, max)
- Temporal features (hour, day, week, month, season)
- Cyclical encoding (sin/cos transformations)
- Holiday or special event indicators
- Normalizing and scaling data

### Phase 5: Train / Validation / Test Split
- 70% Train
- 15% Validation
- 15% Test
- No random shuffling (time-series safe)

### Phase 6–10: Model Building & Training
- Start with baseline models (Naive Baseline, Seasonal Naive)
- ARIMA/ Prophet for quick baseline
- Implement LSTM for deep learning
- Ensemble multiple models (XG Boost, LightGBM)
- Hyperparameter tuning

### Phase 11: Model Comparison
- Rank models by MAE
- Visualize performance metrics

### Phase 12: Reporting
- Generate final performance summary
- Save all plots automatically
 <img width="5369" height="1759" alt="08_model_comparison" src="https://github.com/user-attachments/assets/acc23601-667d-421b-b265-dacf8a483e40" />
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
LSTM gives lowest values for all three metrics.

---

## Outcomes:

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

## Why I chose LSTM? (Explanation)

1. Massive available dataset with 52966 values, high non-linearity, complex long term dependencies, and need for high accuracy for as complex domain as energy prediction, LSTM is highly suited. 
2. LSTM networks manually learn relevant features from raw sequential data, eliminating the need for manual, time-consuming feature engineering (like computing moving averages or seasonality checks) required by classical methods.
3. LSTM can handle raw , non-stationary, and non-linear patterns for real world data like energy prediction, weather forecasting, etc.
4. Traditional models are better suited for linear patterns and statistical analysis. LSTM which is a deep learning RNN model is widely used for non-linear, complex data.
5. Traditional models suffer from vanishing gradients, so they likely don't remeber information from long sequences. LSTM has special gating mechanism which allows it to retain information over longer sequence. This ir crucial for hourly, daily or weekly energy prediction.
Here is the output for daily energy prediction:
<img width="5370" height="2966" alt="11_test_performance_vs_future_forecast" src="https://github.com/user-attachments/assets/371ef1c2-69b5-4d3f-aa55-59f17cd0ceee" />

