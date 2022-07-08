from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('data.txt')
target = data.loc[:, "Y"]
data = data.drop(["Y"], axis=1)

plt.scatter(data[(target == 0)].X1, data[(
    target == 0)].X2, c="red", marker='o')
plt.scatter(data[(target == 1)].X1, data[(
    target == 1)].X2, c="blue", marker='o')
plt.show()

#DIAXWRISMOI HIERARCHICAL
#Single linkage kai n_clusters = 2 -> AKYROS
clustering = AgglomerativeClustering(n_clusters=2, linkage='single').fit(data)
#Ypologismos accuracy
print('Accuracy for single link, n_clusters=2 : ',
      accuracy_score(target, clustering.labels_))
# #SXEDIASMOS DEDOMENWN ME DIAFORETIKA XRWMATA
# plt.scatter(data.X1, data.X2, c=clustering.labels_, cmap="bwr")

# plt.show()

# #Complete linkage kai n_clusters = 2 -> KANEI
clustering = AgglomerativeClustering(
    n_clusters=2, linkage='complete').fit(data)
print('Accuracy for complete link, n_clusters=2 : ',
      accuracy_score(target, clustering.labels_))
# #SXEDIASMOS DEDOMENWN ME DIAFORETIKA XRWMATA
plt.scatter(data.X1, data.X2, c=clustering.labels_, cmap="bwr")
plt.show()
#DBSCAN minPts=5 kai eps = 0.75, 1 , 1.25, 1.5  wtf need to manipulate data in other space
for e in [0.75, 1, 1.25, 1.5]:
    clustering = DBSCAN(eps=e, min_samples=5).fit(data)
    clusters = clustering.labels_
    print("Clusters: ", clusters)
    plt.scatter(data.X1, data.X2, c=clusters, cmap="spring")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


#kmeans me n_clusters = 2
kmeans = KMeans(n_clusters=2).fit(data)
plt.scatter(data.X1, data.X2, c=kmeans.labels_, cmap='bwr')
plt.show()
