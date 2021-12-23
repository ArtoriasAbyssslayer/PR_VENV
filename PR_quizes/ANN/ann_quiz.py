import pandas as pd

X1 = [2,2,-2,-2,1,1,-1,-1]
X2 = [2,-2,-2,2,1,-1,-1,1]
Y = [1,1,1,1,2,2,2,2]

alldata = pd.DataFrame({"X1":X1,"X2":X2,"Y":Y})


x = alldata.loc[:,["X1","X2"]]
y = alldata.loc[:,"Y"]


from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(hidden_layer_sizes=(5,20), max_iter=10000)
clf = clf.fit(x,y)
pred = clf.predict(x)

#calculate error
trainingError = [(t-p) for (t,p) in zip(y,pred)]

import numpy as np
#MeanAbsoluteError
MAE = np.mean(np.abs(trainingError))
print(MAE)