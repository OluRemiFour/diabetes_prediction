# customer segmentation using k-means clustering

from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

customer_data = pd.read_csv('mall_customers.csv')

print(customer_data.shape)

# select Annual Income and Spending score
X = customer_data.iloc[:, [3, 4]].values
print(X)

# - - - - -  choosing the number of clusters - - - - - 
# WCSS means = Within Clusters Sum of Squares
# finding wcss value for different number of clusters
wcss = []

for i in range(1, 11): #loop from 1 - 10, looping for 10x
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
# plot an elbow graph
sns.set()
plt.plot(range(1,11), wcss)
plt.title("The Elbow Point Graph")
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
# plt.show()

# Training the k-means clusters model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster