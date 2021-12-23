import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
import  random



#2d array declaration and initialization
rows, cols = (5, 5)
arr = [[random.Random(0,15) for i in range(cols)] for j in range(rows)]
print(arr)
rows, cols = (5, 5)
arr2 = [[random.Random(1,22) for i in range(cols)] for j in range(rows)]
print(arr2)

for i in range(rows):
    for j in range(cols):
        mult =  arr1[i][j]*arr2[i][j]
