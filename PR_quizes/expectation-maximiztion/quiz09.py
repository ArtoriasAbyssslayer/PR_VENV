from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
kmdata = pd.read_csv('quizdata.txt')
# print(kmdata)

target = kmdata.loc[:, "Y"]
kmdata = kmdata.drop(["Y"], axis=1)

# SCEDIASMOS DEDOMENWN ME ENA XRWMA
plt.scatter(kmdata.loc[:, "X1"], kmdata.loc[:, "X2"])
plt.show()

# SXEDIASMOS DEDOMENWN ME DIAFORETIKA XRWMATA ANA KLASH
plt.scatter(kmdata[(target == 1)].X1,
            kmdata[(target == 1)].X2, c='yellow', marker='o')
plt.scatter(kmdata[(target == 2)].X1, kmdata[(
    target == 2)].X2, c='blue', marker='o')
plt.scatter(kmdata[(target == 3)].X1,
            kmdata[(target == 3)].X2, c='green', marker='o')
plt.show()

# OMADOPOIHSH SE 3 CLUSTERS ME KMEANS SXEDIASMOS CLUSTERS ME TA CENTERS
kmeans = KMeans(n_clusters=3).fit(kmdata)

plt.scatter(kmdata.loc[:, "X1"], kmdata.loc[:, "X2"], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], marker='+', s=169, c=range(3))
plt.show()

# OMADOPOIHSH SE 3 CLUSTERS ME GAUSSIAN MIXTURE ME EPSILON=0.0001 KAI SXEDIASMOS CLUSTERS ME TA CENTERS
gm = GaussianMixture(n_components=3, tol=0.0001).fit(kmdata)


plt.scatter(kmdata.loc[:, "X1"], kmdata.loc[:, "X2"], c=gm.predict(kmdata))
plt.scatter(gm.means_[:, 0], gm.means_[:, 1], marker='+', s=169, c=range(3))
plt.show()
