import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and explore datasets
data_df = pd.read_csv('dataset/Data.csv')
stock_df = pd.read_csv('dataset/StockPrice.csv')

# Convert dates and merge
data_df['Date'] = pd.to_datetime(data_df['Date'])
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
merged_df = pd.merge(data_df.sort_values('Date'), 
                     stock_df.sort_values('Date'), 
                     on='Date', how='inner')

# Visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Data variable over time
plt.subplot(3, 2, 1)
plt.plot(merged_df['Date'], merged_df['Data'], linewidth=0.8)
plt.title('Data Variable Over Time')
plt.xlabel('Date')
plt.ylabel('Data Value')
plt.xticks(rotation=45)

# Plot 2: Stock Price over time
plt.subplot(3, 2, 2)
plt.plot(merged_df['Date'], merged_df['Price'], linewidth=0.8, color='orange')
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)

# Plot 3: Correlation heatmap
plt.subplot(3, 2, 3)
correlation = merged_df[['Data', 'Price']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')

# Plot 4: Scatter plot
plt.subplot(3, 2, 4)
plt.scatter(merged_df['Data'], merged_df['Price'], alpha=0.5, s=1)
plt.title('Data vs Price Relationship')
plt.xlabel('Data')
plt.ylabel('Price')

# Plot 5: Distribution of Data
plt.subplot(3, 2, 5)
plt.hist(merged_df['Data'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Distribution of Data Variable')
plt.xlabel('Data Value')
plt.ylabel('Frequency')

# Plot 6: Distribution of Price
plt.subplot(3, 2, 6)
plt.hist(merged_df['Price'], bins=50, edgecolor='black', alpha=0.7, color='orange')
plt.title('Distribution of Stock Price')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('EDA_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical Summary
print("="*80)
print("EXPLORATORY DATA ANALYSIS SUMMARY")
print("="*80)
print(f"\nDataset Shape: {merged_df.shape}")
print(f"Date Range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
print(f"\nData Variable Statistics:\n{merged_df['Data'].describe()}")
print(f"\nStock Price Statistics:\n{merged_df['Price'].describe()}")
print(f"\nCorrelation: {merged_df['Data'].corr(merged_df['Price']):.4f}")
