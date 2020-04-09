# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:15:45 2020

@author: LO
"""
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

file = open("SP500.pickle","rb")
backup_list = pickle.load(file)
file.close()


changes = np.zeros((468,1257))

for i in range(468):
    for j in range(1257):
        if(np.isnan(backup_list[i]['Close'][j])):
            backup_list[i]['Close'][j]= 0
        if(np.isnan(backup_list[i]['Open'][j])):
            backup_list[i]['Open'][j]= 0
        if(np.isnan(backup_list[i]['Volume'][j])):
            backup_list[i]['Volume'][j]= 0
            
            
        stock_close = backup_list[i]['Close'][j]
        stock_open = backup_list[i]['Open'][j]
        stock_volume = backup_list[i]['Volume'][j]
        changes[i][j] = stock_volume#每天每一家公司的close-open

changes -= changes[64]


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()#正規劃
new = normalizer.fit_transform(changes)


#from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
#normalizer = Normalizer()

#kmeans = KMeans(n_clusters=10, max_iter=1000)
#
#pipeline = make_pipeline(normalizer,kmeans)
#pipeline.fit(changes)

companies = []
for i in range(468):
    companies.append(i)
#labels = pipeline.predict(changes)
#df = pd.DataFrame({'labels': labels,'comapnies':companies })
#print(df.sort_values('labels'))


from sklearn.decomposition import PCA 

reduced_data = PCA(n_components = 2).fit_transform(new)#用PCA降維度以方便可視化


kmeans = KMeans(n_clusters=3)
kmeans.fit(reduced_data)
labels = kmeans.predict(reduced_data)#標籤

df = pd.DataFrame({'labels': labels, 'companies': companies})

print(df.sort_values('labels'))



h = 0.01
#畫邊界
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:,0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#獲取標籤
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap = plt.cm.Paired

# plot figure
plt.clf()
plt.figure(figsize=(10,10))
plt.imshow(Z, interpolation='nearest',
 extent = (xx.min(), xx.max(), yy.min(), yy.max()),
 cmap = cmap,
 aspect = 'auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=5)

#中心點畫白色叉叉
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
 marker='x', s=169, linewidth=3,
 color='w', zorder=10)

plt.title('K-Means Clustering on Stock Market Movements (PCA-Reduced Data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()


A = []
for i in range(len(df.index[labels])):
    
    A.append(df.index[labels][i])
    
df2 = df.sort_values(by = 'labels')
B = []
for i in range(3):
    B.append(A.count(i))
    
#for i in range(len(B)):
#    for j in range(B[i]):
#        plt.plot(changes[df2.index[j]])
#    plt.show()
