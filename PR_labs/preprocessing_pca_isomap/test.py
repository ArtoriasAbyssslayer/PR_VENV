import pandas as pd
import matplotlib.pyplot as plt
# x = [1,0,-1,0,-1,1]
# y = [0,1,1,-1,0,-1]
# z = [-1,-1,0,1,1,0]
#
# labels = ["x1", "x2","x3", "x4","x5", "x6"]
#
# pdata = pd.DataFrame({"X":x,"Y":y,"Z":z}, index=labels)
#
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.scatter(pdata.X, pdata.Y, pdata.Z)
#
# # ploting data in 3d space although it is 6d data_sample
# #plot in 3d space
#
# for i in range(len(pdata.index)):
#     ax.text(pdata.loc[labels[i], "X"], pdata.loc[labels[i], "Y"], pdata.loc[labels[i], "Z"], '%s' % (str(labels[i])), size=20, zorder=1)
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# plt.show()

engdata = pd.read_csv('./engdata.txt')
print(engdata)

pdata = engdata.loc[:,"Age", "Salary"] #select some columnes
pdata = pdata.drop_duplicates() #drop drop_duplicates
print(pdata)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
