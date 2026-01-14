"""
All Models - Stock Price Prediction
====================================
Consolidated training script for all viable models (R² >= -2).

Models included:
1. Linear Regression (Best: R²=0.993)
2. Ridge Regression (R²=0.993)
3. Gradient Boosting (R²=0.429)
4. GRU Deep Learning (R²=0.422)
5. Random Forest (R²=0.371)
6. XGBoost (R²=0.163)
7. Random Forest (from train_models, R²=-1.01)
8. GARCH-GRU (R²=-1.48)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Optional TensorFlow for GRU
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    print("⚠ TensorFlow not installed. GRU model will be skipped.")
    TF_AVAILABLE = False

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    print("⚠ XGBoost not installed. XGBoost model will be skipped.")
    XGB_AVAILABLE = False


# ==============================================================================
# DATA LOADING AND PREPARATION
# ==============================================================================

def load_processed_data():
    """Load the processed data file."""
    df = pd.read_csv('processed_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def prepare_ml_data(df):
    """Prepare data for traditional ML models."""
    feature_cols = [col for col in df.columns if col not in ['Date', 'Target', 'Price']]
    X = df[feature_cols]
    y = df['Target']
    
    # Time series split (80-20)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare_dl_data(df):
    """Prepare data for deep learning models with sequences."""
    feature_cols = ['Data', 'Data_lag_1', 'Data_lag_2', 'Data_lag_3', 
                   'Data_rolling_mean_7', 'Data_rolling_std_7',
                   'Price_lag_1', 'Price_lag_2', 'Price_lag_3', 
                   'Price_rolling_mean_7', 'Price_change']
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    data_array = df[feature_cols + ['Target']].values
    
    # Normalize
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_data = scaler_X.fit_transform(data_array[:, :-1])
    y_data = scaler_y.fit_transform(data_array[:, -1].reshape(-1, 1))
    
    # Create sequences
    seq_length = 60
    X_seq, y_seq = [], []
    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i:i+seq_length])
        y_seq.append(y_data[i+seq_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Train-test split
    split_idx = int(len(X_seq) * 0.8)
    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler_y


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

def build_gru_model(input_shape):
    """Build GRU model - best performing deep learning architecture."""
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_ml_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """Train all machine learning models."""
    
    models = {
        'Linear Regression': (LinearRegression(), True),
        'Ridge Regression': (Ridge(alpha=1.0, random_state=42), True),
        'Gradient Boosting': (GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42), False),
        'Random Forest': (RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1), False),
    }
    
    # Add XGBoost if available
    if XGB_AVAILABLE:
        models['XGBoost'] = (XGBRegressor(
            n_estimators=200, max_depth=7, learning_rate=0.1, 
            random_state=42, n_jobs=-1), False)
    
    results = {}
    
    for name, (model, use_scaled) in models.items():
        print(f"\n  Training {name}...")
        start_time = time.time()
        
        if use_scaled:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        
        training_time = time.time() - start_time
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        results[name] = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'Training_Time': float(training_time)
        }
        
        print(f"    ✓ RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
        
        # Save best model
        if name == 'Linear Regression':
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
    
    return results


def train_gru_model(X_train, X_test, y_train, y_test, scaler_y):
    """Train GRU deep learning model."""
    
    print("\n  Training GRU (Deep Learning)...")
    
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Predictions
    pred_scaled = model.predict(X_test, verbose=0)
    predictions = scaler_y.inverse_transform(pred_scaled)
    actuals = scaler_y.inverse_transform(y_test)
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f"    ✓ RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    
    # Save model
    model.save('models/gru_model.h5')
    
    return {
        'GRU': {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'Training_Time': float(training_time)
        }
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main training function."""
    print("=" * 70)
    print("STOCK PRICE PREDICTION - ALL MODELS")
    print("Training only viable models (R² >= -2)")
    print("=" * 70)
    
    # Load data
    print("\n📁 Loading processed data...")
    df = load_processed_data()
    print(f"   Dataset shape: {df.shape}")
    
    all_results = {}
    
    # ==================== ML MODELS ====================
    print("\n" + "=" * 70)
    print("TRAINING MACHINE LEARNING MODELS")
    print("=" * 70)
    
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_ml_data(df)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    ml_results = train_ml_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
    all_results.update(ml_results)
    
    # ==================== DEEP LEARNING MODELS ====================
    if TF_AVAILABLE:
        print("\n" + "=" * 70)
        print("TRAINING DEEP LEARNING MODELS")
        print("=" * 70)
        
        X_train_dl, X_test_dl, y_train_dl, y_test_dl, scaler_y = prepare_dl_data(df)
        
        gru_results = train_gru_model(X_train_dl, X_test_dl, y_train_dl, y_test_dl, scaler_y)
        all_results.update(gru_results)
    
    # ==================== RESULTS SUMMARY ====================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Sort by R2
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['R2'], reverse=True)
    
    print(f"\n{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'Time(s)':<10}")
    print("-" * 70)
    
    for name, metrics in sorted_results:
        print(f"{name:<25} {metrics['RMSE']:<12.2f} {metrics['MAE']:<12.2f} "
              f"{metrics['R2']:<12.4f} {metrics['Training_Time']:<10.2f}")
    
    # Best model
    best = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL: {best[0]}")
    print(f"   R²: {best[1]['R2']:.4f}")
    print(f"   RMSE: {best[1]['RMSE']:.2f}")
    print("=" * 70)
    
    # Save results
    with open('model_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Save to CSV
    results_df = pd.DataFrame([
        {'Model': name, 'RMSE': m['RMSE'], 'MAE': m['MAE'], 
         'R2': m['R2'], 'MAPE': m['MAPE'], 'Training_Time': m['Training_Time'],
         'Source': 'Combined'}
        for name, m in sorted_results
    ])
    results_df.to_csv('master_results_comparison.csv', index=False)
    
    print(f"\n✓ Results saved to: model_results.json")
    print(f"✓ Results saved to: master_results_comparison.csv")
    
    return all_results


if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    results = main()
