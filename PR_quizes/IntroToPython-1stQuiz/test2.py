import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os


# import dataset
data = pd.read_csv('./quiz2_data.csv',sep='.')
print(data)
