import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Data Collection Simulation (Logistics Dataset)
raw_data = {
    'shipment_id': [101, 102, 103, 104, 105, 106],
    'distance_km': [150.0, 300.0, np.nan, 450.0, 1200.0, 200.0],  # Missing value + Outlier (1200)
    'shipping_cost': [2500.0, 4800.0, 3100.0, np.nan, 18000.0, 3000.0],
    'delivery_days': [2, 4, 3, 5, 15, 2],  # Outlier (15 days)
    'carrier_type': ['Express', 'Standard', 'Standard', 'Express', np.nan, 'Express']
}

df = pd.DataFrame(raw_data)
print("--- Raw Logistics Data ---")
print(df, "\n")

# 2. Handling Missing Values
# Numeric columns: Impute with Median
df['distance_km'].fillna(df['distance_km'].median(), inplace=True)
df['shipping_cost'].fillna(df['shipping_cost'].median(), inplace=True)

# Categorical column: Impute with Mode
df['carrier_type'].fillna(df['carrier_type'].mode()[0], inplace=True)

# 3. Outlier Detection & Capping using IQR Method
def cap_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, 
                 np.where(df[column] < lower_bound, lower_bound, df[column]))

cap_outliers('distance_km')
cap_outliers('shipping_cost')

# 4. Data Normalization / Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['distance_km', 'shipping_cost', 'delivery_days']])
df_scaled = pd.DataFrame(scaled_features, columns=['distance_km_scaled', 'shipping_cost_scaled', 'delivery_days_scaled'])

# Final Processed Dataset
df_final = pd.concat([df[['shipment_id', 'carrier_type']], df_scaled], axis=1)
print("--- Cleaned and Processed Data ---")
print(df_final)
