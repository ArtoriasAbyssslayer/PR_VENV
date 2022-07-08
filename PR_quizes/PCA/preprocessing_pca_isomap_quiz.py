from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
data = pd.read_csv("./quiz_data.csv", sep=',')
print(data)

trainingRange = list(range(0, 50)) + list(range(90, 146))
training = data.loc[trainingRange, :]
trainingType = training.loc[:, "Type"]
training = training.drop(["Type"], axis=1)
print("training", training)


testingRange = list(range(50, 90))
testing = data.loc[testingRange, :]
testingType = testing.loc[:, "Type"]
testing = testing.drop(["Type"], axis=1)
print("Testing", testing)


# find the
scaler = StandardScaler()
scaler = scaler.fit(training)
transformed = pd.DataFrame(scaler.transform(
    training), columns=training.columns)
pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
plt.title("PCA1,PCA2,PCA3...etc comparinson with ratio")
plt.show()

print(eigenvalues[0]/sum(eigenvalues))

pca = PCA(n_components=4)
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
#inverse the algoritm
pca_inverse = pd.DataFrame(pca.inverse_transform(
    pca_transformed), columns=training.columns)
info_loss = (eigenvalues[4]+eigenvalues[5]
             + eigenvalues[6]+eigenvalues[7] + eigenvalues[8]) / sum(eigenvalues)
print(info_loss)

scaler = StandardScaler()
scaler = scaler.fit(testing)
transformed_testing = pd.DataFrame(scaler.transform(
    testing), columns=training.columns)
pca_transformed = pca.transform(transformed)
#pca last
#accuracy scores matrix
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(training, trainingType)
preds = clf.predict(testing)
acc = accuracy_score(testingType, preds)
print("acc:", acc)
recall = recall_score(training, preds)
print("recall", recall)
accs = []
for i in range(len(eigenvalues)):
    pca = PCA(n_components=i+1)
    pca = pca.fit(transformed)
    train_pca = pca.transform(transformed)
    test_pca = pca.transform(transformed_testing)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(train_pca, trainingType)
    preds = clf.predict(test_pca)
    accs.append(accuracy_score(testingType, preds))

print(accs)
