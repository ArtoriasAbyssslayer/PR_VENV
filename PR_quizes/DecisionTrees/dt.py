import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import tree



#load iris dataset

iris = datasets.load_iris()
variable_index = iris.feature_names.index("petal length (cm)")

#load data
data = iris.data

#create iris dataFrame

df = pd.DataFrame(data, columns=iris.feature_names)
