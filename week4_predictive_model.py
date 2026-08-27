import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Dataset Simulation for Predictive Modeling
np.random.seed(42)
n_samples = 200

data = {
    'distance_km': np.random.uniform(10, 500, n_samples),
    'package_weight_kg': np.random.uniform(1, 50, n_samples),
    'traffic_delay_index': np.random.uniform(1, 5, n_samples),
    'weather_factor': np.random.uniform(1, 3, n_samples),
    'driver_experience_years': np.random.randint(1, 15, n_samples)
}

df = pd.DataFrame(data)

# Target Variable: Delivery Time in hours (created with some realistic relationship + noise)
df['delivery_time_hours'] = (
    (df['distance_km'] / 50.0) +
    (df['traffic_delay_index'] * 0.8) +
    (df['weather_factor'] * 0.5) -
    (df['driver_experience_years'] * 0.1) +
    np.random.normal(0, 0.5, n_samples)
)

# 2. Features and Target Selection
X = df[['distance_km', 'package_weight_kg', 'traffic_delay_index', 'weather_factor', 'driver_experience_years']]
y = df['delivery_time_hours']

# Split data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training - Linear Regression vs Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)

# 4. Model Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("--- Model Performance Metrics (Random Forest) ---")
print(f"MAE  (Mean Absolute Error) : {round(mae, 3)} hours")
print(f"RMSE (Root Mean Squared Error): {round(rmse, 3)} hours")
print(f"R² Score                   : {round(r2, 3)}")

# 5. Prediction Example for Route Optimization
sample_route = [[250, 15, 3.5, 1.2, 5]]  # 250km, 15kg, moderate traffic & weather
predicted_hours = model_rf.predict(sample_route)
print(f"\nPredicted Delivery Time for new route: {round(predicted_hours[0], 2)} hours")
