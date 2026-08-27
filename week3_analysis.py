import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for charts
sns.set_theme(style="whitegrid")

# 1. Logistics Dataset Simulation
np.random.seed(42)
n_shipments = 100

data = {
    'shipment_id': range(1001, 1001 + n_shipments),
    'distance_km': np.random.uniform(50, 800, n_shipments),
    'shipment_weight_kg': np.random.uniform(10, 500, n_shipments),
    'transport_cost': np.random.uniform(1000, 15000, n_shipments),
    'delivery_days': np.random.randint(1, 8, n_shipments),
    'transport_mode': np.random.choice(['Road', 'Air', 'Rail', 'Sea'], n_shipments),
    'delivery_status': np.random.choice(['On-Time', 'Delayed'], n_shipments, p=[0.75, 0.25])
}

df = pd.DataFrame(data)

# 2. Summary Statistics (EDA)
print("--- Summary Statistics ---")
print(df[['distance_km', 'transport_cost', 'delivery_days']].describe())

# 3. Visualization 1: Cost vs Distance by Transport Mode (Scatter Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='distance_km', y='transport_cost', hue='transport_mode', style='delivery_status', s=80)
plt.title('Transportation Cost vs Distance (by Mode)')
plt.xlabel('Distance (km)')
plt.ylabel('Transport Cost (₹)')
plt.tight_layout()
plt.savefig('cost_vs_distance.png')
plt.show()

# 4. Visualization 2: Delivery Days Distribution (Boxplot)
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='transport_mode', y='delivery_days', palette='Set2')
plt.title('Delivery Duration across Transport Modes')
plt.xlabel('Transport Mode')
plt.ylabel('Delivery Days')
plt.tight_layout()
plt.savefig('delivery_time_boxplot.png')
plt.show()

# 5. Visualization 3: Correlation Heatmap
plt.figure(figsize=(6, 4))
numeric_df = df[['distance_km', 'shipment_weight_kg', 'transport_cost', 'delivery_days']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Logistics Metrics Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
