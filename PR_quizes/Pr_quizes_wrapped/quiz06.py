import pandas as pd
data = pd.read_csv('quiz_data.csv', sep=',')

trainingRange = list(range(0,50)) + list(range(90,146))
training = data.loc[trainingRange, :]
trainingType = training.loc[:,'Type']
training = training.drop(['Type'], axis=1)

testingRange = list(range(50,90))
testing = data.loc[testingRange, :]
testingType = testing.loc[:,'Type']
testing = testing.drop(['Type'], axis=1)

#efarmogh scaling kai pca sta training data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(training)
transformed = pd.DataFrame(scaler.transform(training), columns=training.columns)

pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

#pososto plhroforias gia ta pc
import matplotlib.pyplot as plt
plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
plt.show()
print('Pososto plhroforias gia to pc1: ', eigenvalues[0]/sum(eigenvalues))

#pososto apwleias plhroforias an krathsoume mono ta 4 prwta pc
n = len(eigenvalues)
lost = 0
keep = 4
for i in range(4,n):
    lost = lost + eigenvalues[i]
info_loss = lost / sum(eigenvalues)
print('Info Loss: ',info_loss)

#KNN MODELO
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(training, trainingType)
pred = clf.predict(testing)

from sklearn.metrics import accuracy_score, recall_score
print("Accuracy: ", accuracy_score(testingType, pred))
print('Recall: ', recall_score(testingType,pred,pos_label=2))




acc=[]
for i in range (1,10):
    # Scaler according to training data
    scaler = StandardScaler()
    scaler = scaler.fit(training)
    # Apply scalling at training and testing data
    scaled_training = pd.DataFrame(scaler.transform(training), columns=training.columns)
    scaled_testing = pd.DataFrame(scaler.transform(testing), columns=testing.columns)
    # PCA according to training data
    pca = PCA(n_components=i)
    pca = pca.fit(scaled_training)
    # Apply PCA at training and testing data
    pca_training = pd.DataFrame(pca.transform(scaled_training))
    pca_testing = pd.DataFrame(pca.transform(scaled_testing))
    # Train KNN with training data after PCA
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(pca_training, trainingType)
    # Predict with testing data
    pred = clf.predict(pca_testing)
    acc.append(accuracy_score(testingType, pred))


print("Accuracy: ", acc)