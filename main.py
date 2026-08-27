import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data create kar rahe hain testing ke liye
sample_data = {
    'actual_delivery': [3, 2, 5, 4, 1, 3, 2],
    'expected_delivery': [3, 3, 4, 4, 2, 3, 2],
    'transport_cost': [150, 200, 300, 250, 100, 180, 220],
    'items_sold': [500, 600, 450, 700, 300, 550, 620],
    'average_inventory': [100, 120, 90, 110, 80, 105, 115],
    'latitude': [19.07, 19.08, 19.05, 19.09, 19.01, 19.06, 19.04],
    'longitude': [72.87, 72.88, 72.85, 72.89, 72.81, 72.86, 72.84],
    'distance': [10, 15, 25, 20, 5, 12, 18],
    'traffic_index': [2, 4, 5, 3, 1, 3, 4],
    'delivery_time': [1.5, 2.5, 4.0, 3.0, 1.0, 2.0, 2.8]
}
data = pd.DataFrame(sample_data)

# KPI 1: Delivery Time Accuracy
on_time = (data['actual_delivery'] <= data['expected_delivery']).mean()
print(f"Delivery Time Accuracy: {round(on_time * 100, 2)}%")

# KPI 2: Transportation Cost Efficiency
avg_cost = data['transport_cost'].mean()
print(f"Average Transportation Cost per Delivery: ₹{round(avg_cost, 2)}")

# KPI 3: Inventory Turnover Ratio
turnover_ratio = data['items_sold'].sum() / data['average_inventory'].mean()
print(f"Inventory Turnover Ratio: {round(turnover_ratio, 2)}")

# Route Optimization using Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['route_cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])
print("\nRoute clusters assigned successfully.")

# Predictive Modeling for Delivery Time
X = data[['distance', 'traffic_index']]
y = data['delivery_time']
model = LinearRegression().fit(X, y)
predicted_time = model.predict([[12, 3]])
print(f"Predicted Delivery Time for 12 km distance & traffic index 3: {round(predicted_time[0], 2)} hours")

# Visualization
sns.scatterplot(x='distance', y='delivery_time', hue='route_cluster', data=data)
plt.title("Delivery Time vs Distance (Clustered Routes)")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (hours)")
plt.show()
