import numpy as np
import pandas as pd
from sklearn import datasets as sklearn_datasets




iris = sklearn_datasets.load_iris()
featureIndex = iris.feature_names.index('sepal length (cm)')
data = iris.data
print(featureIndex)
print(np.mean(data, axis=0))
