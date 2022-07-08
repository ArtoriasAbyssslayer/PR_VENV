from sklearn.metrics import silhouette_samples, silhouette_score
import math
from sklearn.cluster import KMeans
import pandas as pd
data = pd.read_csv('quiz_data.csv')
#print(data)
# initial = [[-4, 10],
#            [0, 0],
#            [4, 10]]
initialPointPairs = pd.DataFrame({"Xo": [-4, 0, 4], "Yo": [10, 0, 10]})
kmeans = KMeans(n_clusters=3, init=initialPointPairs).fit(data)
#cohesion
print("Cohesion : ", kmeans.inertia_)

#separation
separation = 0


def distance(x1, x2): return math.sqrt(
    ((x1.X1 - x2.X1)**2) + ((x1.X2 - x2.X2)**2))


m = data.mean()
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_ == i, :].mean()
    Ci = len(data.loc[kmeans.labels_ == i, :].index)
    separation += Ci * (distance(m, mi)**2)

print("Separation: ", separation)

#silhouette
# meso silhouette gia thn omadopoihsh
print("Meso silhouette: ", silhouette_score(data, kmeans.labels_))

#---------------2o clustering----------------
initial = pd.DataFrame({"Xo": [-2, 2, 0], "Yo": [0, 0, 10]})
kmeans = KMeans(n_clusters=3, init=initial).fit(data)
#cohesion
print("Cohesion : ", kmeans.inertia_)

#separation
separation = 0


def distance(x1, x2): return math.sqrt(
    ((x1.X1 - x2.X1)**2) + ((x1.X2 - x2.X2)**2))


m = data.mean()
for i in list(set(kmeans.labels_)):
    mi = data.loc[kmeans.labels_ == i, :].mean()
    Ci = len(data.loc[kmeans.labels_ == i, :].index)
    separation += Ci * (distance(m, mi)**2)

print("Separation: ", separation)

#silhouette

print("Average silhouette: ", silhouette_score(data, kmeans.labels_))
