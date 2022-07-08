bdata = data.loc[:, [""]]

sse = []
for i in range(1, 11):
    sse.append(
        KMeans(n_clusters=i, init=bdata.loc[0:i-1, :]).fit(bdata).inertia_)
plt.plot(range(1, 11), sse)
plt.scatter(range(1, 11), sse, marker="o")
plt.show()
