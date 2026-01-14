import pandas as pd
import numpy as np

def create_features(df):
    """
    Comprehensive feature engineering for stock price prediction
    """
    df = df.copy()
    
    # Lag features for Data
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df[f'Data_lag_{lag}'] = df['Data'].shift(lag)
    
    # Change features
    df['Data_change'] = df['Data'].diff()
    df['Data_pct_change'] = df['Data'].pct_change()
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'Data_rolling_mean_{window}'] = df['Data'].rolling(window=window).mean()
        df[f'Data_rolling_std_{window}'] = df['Data'].rolling(window=window).std()
        df[f'Data_rolling_min_{window}'] = df['Data'].rolling(window=window).min()
        df[f'Data_rolling_max_{window}'] = df['Data'].rolling(window=window).max()
    
    # Momentum features
    df['Data_momentum_3'] = df['Data'] - df['Data'].shift(3)
    df['Data_momentum_7'] = df['Data'] - df['Data'].shift(7)
    
    # Rate of Change
    for period in [7, 14, 30]:
        df[f'Data_roc_{period}'] = ((df['Data'] - df['Data'].shift(period)) / 
                                     df['Data'].shift(period) * 100)
    
    # Price lag features
    for lag in [1, 2, 3, 5, 7]:
        df[f'Price_lag_{lag}'] = df['Price'].shift(lag)
    
    # Price change features
    df['Price_change'] = df['Price'].diff()
    df['Price_pct_change'] = df['Price'].pct_change()
    
    # Price rolling statistics
    for window in [7, 14, 30]:
        df[f'Price_rolling_mean_{window}'] = df['Price'].rolling(window=window).mean()
        df[f'Price_rolling_std_{window}'] = df['Price'].rolling(window=window).std()
    
    # Volatility measures
    df['Price_volatility_7'] = (df['Price'].rolling(window=7).std() / 
                                 df['Price'].rolling(window=7).mean())
    df['Price_volatility_30'] = (df['Price'].rolling(window=30).std() / 
                                  df['Price'].rolling(window=30).mean())
    
    # Time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Target variable (next day's price)
    df['Target'] = df['Price'].shift(-1)
    
    return df

# Load and process data
data_df = pd.read_csv('dataset/Data.csv')
stock_df = pd.read_csv('dataset/StockPrice.csv')

data_df['Date'] = pd.to_datetime(data_df['Date'])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

merged_df = pd.merge(data_df.sort_values('Date'), 
                     stock_df.sort_values('Date'), 
                     on='Date', how='inner')

# Apply feature engineering
df_featured = create_features(merged_df)

# Remove NaN values
df_clean = df_featured.dropna().reset_index(drop=True)

# Save processed data
df_clean.to_csv('processed_data.csv', index=False)

print(f"Features created: {df_clean.shape[1]}")
print(f"Clean dataset shape: {df_clean.shape}")
print(f"Saved to: processed_data.csv")
