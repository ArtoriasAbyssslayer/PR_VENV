# Import Libraries (genika kalo na uparxoun ola mesa just in case)

from numpy.core.fromnumeric import ravel, var
import pandas as pd
import numpy as np
import statistics as stat
from sklearn import datasets
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# ------------ QUIZ 1 ------------

# iris data set load
iris = datasets.load_iris()
featureIndex = iris.feature_names.index("petal length (cm)") # gia oti column zhtaei vazoume to antistoix
data = iris.data

# Dhmiourgia DataFrame gia dedomena
df = pd.DataFrame(data, columns=iris.feature_names)


# Mesos oros (mpakalikos alla polu praktikos tropos)
mo = stat.mean(data[:, 2]) # 2 shmainei 3rd column ston pinaka

# Mesos oros (me Dataframe use)
mo = df[["petal length (cm)"]].mean()

# Megisth timh (mpakalikos alla praktikos tropos)
megisto = max(data[:, 1]) # 1 shmainei 2νδ column ston pinaka

# Megisth timh (me DataFrame)
megisto = df[["sepal width (cm)"]].max()

# Variance (mpakalikos alla praktikos tropos)
diakumansh = stat.variance(data[:,0]) # 0 shmainei 1h sthlh tou pinaka

# Variance (me Dataframe)
diakumansh = df[["sepal length (cm)"]].var()

# Mesos oros kathe sthlhs se vector me strogullopoihsh 2ou dekadikou
mo_all = df.mean()
mo_all = round(mo_all, 2)



# ------------ QUIZ 2 ------------

# Read File
market = pd.read_csv("./quiz2.csv")


# Ypologismos Synolikous GINI me xrhsh Tree (ousiastika einai to prwto GINI gia ton prwto diaxwrismo)
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder.fit(market.loc[:, ["Sex", "CarType", "Budget"]])
transformedCarType = encoder.transform(market.loc[:, ["Sex", "CarType", "Budget"]])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(transformedCarType, market.loc[:, "Insurance"])
# fig = plt.figure(figsize=(8,7))
# tree.plot_tree(clf, class_names=["No","Yes"], filled=True)
# plt.show()

# Ypologismos GINI gia to xarakthristiko Male
freq = pd.crosstab(market.Sex , market.Insurance, normalize="index") # ypologismos suxnothtwn
freqSum = pd.crosstab(market.Sex, market.Insurance, normalize="all").sum(axis=1,'columns'=2) # ypologismos sunolikwn syxnwthtwn
GINI_Male = 1 - freq.loc["M", "No"] ** 2 - freq.loc["M", "Yes"] ** 2
GINI_Female = 1 - freq.loc["F", "No"] ** 2 - freq.loc["F", "Yes"] ** 2
GINI_Sex = freqSum["M"] * GINI_Male + freqSum["F"] * GINI_Female


# Ypologismos GINI CarType MultiWay Split
absfreq = pd.crosstab(market.CarType, market.Insurance)
freq = pd.crosstab(market.CarType, market.Insurance, normalize='index')
freqSum = pd.crosstab(market.CarType, market.Insurance, normalize='all').sum(axis=1)
GINI_Family = 1 - freq.loc["Family", "No"]**2 - freq.loc["Family", "Yes"]**2
GINI_Sport = 1 - freq.loc["Sport", "No"]**2 - freq.loc["Sport", "Yes"]**2
GINI_Sedan = 1 -freq.loc["Sedan", "No"]**2 - freq.loc["Sedan", "Yes"]**2
GINI_CarType = freqSum.loc["Family"] * GINI_Family + freqSum["Sport"] * GINI_Sport + freqSum["Sedan"] * GINI_Sedan


# Ypologismos GINI Budget MultiWay Split
freq = pd.crosstab(market.Budget, market.Insurance, normalize="index")
freqSum = pd.crosstab(market.Budget, market.Insurance, normalize="all").sum(axis=1)
GINI_Low = 1 - freq.loc["Low", "No"] ** 2 - freq.loc["Low", "Yes"] ** 2
GINI_Medium = 1 - freq.loc["Medium", "No"] ** 2 - freq.loc["Medium", "Yes"] ** 2
GINI_High = 1 - freq.loc["High", "No"] ** 2 - freq.loc["High", "Yes"] ** 2
GINI_VeryHigh = 1 - freq.loc["VeryHigh", "No"] ** 2 - freq.loc["VeryHigh", "Yes"] ** 2
GINI_Budget = freqSum["Low"] * GINI_Low + freqSum["Medium"] * GINI_Medium + freqSum["High"] * GINI_High + freqSum["VeryHigh"] * GINI_VeryHigh






# ------------ QUIZ 3 ------------

#read tou arxeiou me ta dedomena
data = pd.read_csv("./quiz3.csv")

# diaxwrismos twn stoixeiwn
X = data.loc[:, ["P_M1", "P_M2"]]
y = data.loc[:, ["Class"]]
# print(X)

# Eisagwgh twn dedomenwn se morfh array
predProb = np.array(X)
# print(predProb)

#orismos threshold kai classification analoga me to threshold
th = 0.5
classifcation = predProb >= th
classifcation = classifcation.astype(int) #metatroph Boolean se Int

# Orismos Confusion Matrix gia na paroume ta dedomena tn, tp, fp, fn
tn1, fp1, fn1, tp1 = confusion_matrix(y_true = y.values , y_pred=classifcation[:, 0]).ravel()

# Ypologismos TPR
tpr1 = tp1/(tp1+fn1)
# print(tpr1)

# Ypologismos F_measure
tn2, fp2, fn2, tp2 = confusion_matrix(y_true=y.values, y_pred=classifcation[:, 1]).ravel()
Fmeasure = 2*tp2/(2*tp2 + fn2 + fp2)
# print(Fmeasure)

# Kataskevh ROC kampulhs
fpr1, tpr1, thresholds1 = roc_curve(y, predProb[:, 0])
fpr2, tpr2, thresholds2 = roc_curve(y, predProb[:, 1])
# plt.title('ROC')
# plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % auc(fpr1, tpr1))
# plt.plot(fpr2, tpr2, 'g', label = 'AUC = %0.2f' % auc(fpr2, tpr2))
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()





# ------------ QUIZ 4 ------------

# Eisagwgh dedomenwn se pinakes
X1 = [2, 2, -2, -2, 1, 1, -1, -1]
X2 = [2, -2, -2, 2, 1, -1, -1, 1]
Y = [1, 1, 1, 1, 2, 2, 2, 2]

# Eisagwgh dedomenwn se Dataframe
alldata = pd.DataFrame({"X1":X1, "X2":X2, "Y":Y})

# Diaxwrismos tou DataFrame se X kai Y
X = alldata.loc[:, ["X1", "X2"]]
y = alldata.loc[:, "Y"]


# Neuroniko me 1 HL kai 2HN me max_tier=10000 ( Gia na allaksoume
# layers kai nodes allazoume ta orismata tou hidden_layer_sizes=
# p.x. gia 1 HL kai 2HN vazoume hidden_layer_sizes=(2),
# gia 2HL kai 20HN vazoume hidden_layer_sizes=(20,20)
# Ousiastika to posa stoixeia tha exei h parenthesh einai posa Hidden Layers tha exoume
# kai o arithmos tou kathe stoixeiou einai ta posa Hidden Nodes tha exoume)
clf = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=10000)
clf = clf.fit(X, y)
pred = clf.predict(X)

# Ypologismos Mesou Training Error
trainingError = [(t-p) for (t,p) in zip(y,pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)




# ------------ QUIZ 5 ------------

# X1 = [-2.0, -2.0, -1.8, -1.4, -1.2, 1.2, 1.3, 1.3, 2.0, 2.0, -0.9, -0.5, -0.2, 0.0, 0.0, 0.3, 0.4, 0.5, 0.8, 1.0]
# X2 = [-2.0, 1.0, -1.0, 2.0, 1.2, 1.0, -1.0, 2.0, 0.0, -2.0, 0.0, -1.0, 1.5, 0.0, -0.5, 1.0, 0.0, -1.5, 1.5, 0.0]
# Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

# alldata = pd.DataFrame({"X1" : X1, "X2" : X2, "Y" : Y})

# X = alldata.loc[:, ["X1", "X2"]]
# y = alldata.Y

# plt.scatter(X[(y == 2)].X1, X[(y == 2)].X2, c="red", marker="+")
# plt.scatter(X[(y == 1)].X1, X[(y == 1)].X2, c="blue", marker="o")
# plt.show()

# clf = KNeighborsClassifier(n_neighbors=3)
# clf = clf.fit(X, y)
# print(clf.predict([[1.5, -0.5]]))
# print(clf.predict_proba([[1.5, -0.5]]))

# clf = KNeighborsClassifier(n_neighbors=5)
# clf = clf.fit(X, y)
# print(clf.predict([[-1, 1]]))
# print(clf.predict_proba([[-1, 1]]))

X1 = [2, 2, -2, -2, 1, 1, -1, -1]
X2 = [2, -2, -2, 2, 1, -1, -1, 1]
Y = [1, 1, 1, 1, 2, 2, 2, 2]

alldata = pd.DataFrame({"X1" : X1, "X2" : X2, "Y" : Y})
X = alldata.loc[:, ["X1", "X2"]]
y = alldata.Y



X1_1 = np.arange (min(X.X1.tolist()), max(X.X1.tolist()), 0.01)
X2_2 = np.arange (min(X.X2.tolist()), max(X.X2.tolist()), 0.01)
xx, yy = np.meshgrid(X1_1, X2_2)

clf = svm.SVC(kernel="rbf", gamma=4)
clf = clf.fit(X, y)
pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)
plt.contour(xx, yy, pred, colors="blue")
plt.scatter(X[(y == 2)].X1, X[(y == 2)].X2, c="red", marker="+")
plt.scatter(X[(y == 1)].X1, X[(y == 1)].X2, c="blue", marker="o")
plt.show()

pred = clf.predict(X)
# print(accuracy_score(y, pred))
# print(clf.predict([[-2, 1.9]]))
