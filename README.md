# Stock Price Prediction - Model Comparison

A comprehensive machine learning project comparing various models for stock price prediction, including traditional ML algorithms and deep learning architectures.

## 🏆 Results Summary

| Model | R² Score | RMSE |
|-------|----------|------|
| **Linear Regression** | **0.993** | 51.40 |
| Ridge Regression | 0.993 | 51.44 |
| Gradient Boosting | 0.429 | 473.17 |
| GRU (Deep Learning) | 0.422 | 475.98 |
| Random Forest | 0.371 | 496.33 |
| XGBoost | 0.163 | 572.87 |

> **Key Finding**: Simple Linear Regression outperforms all complex deep learning models with R² = 0.993

## 📁 Project Structure

```
├── all_models.py              # Combined model training script
├── feature_engineering.py     # Feature creation pipeline
├── EDA_notebook.py           # Exploratory data analysis
├── master_model_visualization.py  # Visualization generation
├── master_results_comparison.csv  # Model comparison results
├── dataset/
│   ├── Data.csv              # Input features
│   └── StockPrice.csv        # Target prices
└── models/                   # Saved trained models
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Step 1: Feature Engineering
python feature_engineering.py

# Step 2: Train All Models
python all_models.py

# Step 3: Generate Visualizations
python master_model_visualization.py
```

## 📊 Visualizations

The project generates comprehensive comparison charts:
- `master_model_comparison.png` - Multi-panel dashboard
- `model_ranking_chart.png` - R² score rankings

## 🧪 Models Implemented

### Machine Learning
- Linear Regression
- Ridge Regression  
- Random Forest
- Gradient Boosting
- XGBoost

### Deep Learning
- GRU (Gated Recurrent Unit)

## 📈 Features

The feature engineering pipeline creates:
- Lag features (1-30 days)
- Rolling statistics (mean, std, min, max)
- Momentum indicators
- Rate of change
- Volatility measures
- Time-based features (day, month, quarter)

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- XGBoost
- pandas, numpy, matplotlib, seaborn

## 📄 License

MIT License
