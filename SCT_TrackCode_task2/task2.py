import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset.csv')

# Select features
X = df[['Annual Income', 'Spending Score']]

# KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualization
plt.scatter(X['Annual Income'], X['Spending Score'], c=df['Cluster'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.savefig('customer_segmentation.png')
print("Plot saved as 'customer_segmentation.png'")