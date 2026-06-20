import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
from pca import getnumComp,plotKneeComp
from sklearn.metrics import silhouette_score









      

def getnumClust(dataSet,toll,iter):
    inertia = []
    maxK=100
    for i in range(1, maxK):
        kmeans = KMeans(n_clusters=i,n_init=1,init='k-means++',tol=toll,max_iter=iter,random_state=42)
        kmeans.fit(dataSet)
        inertia.append(kmeans.inertia_)
    
    kl = KneeLocator(range(1, maxK), inertia, curve="convex", direction="decreasing")
    return kl.elbow,inertia


 


def plotKneeClust(y,x,elbow,ylabel,xlabel):
    plt.plot(x, y, 'bx-')
    plt.scatter(elbow, y[elbow - 1], c='red', label=f'best k: {elbow}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Ricerca k ottimale')
    plt.legend()
    plt.show()

def clustering(dataset,elbow,toll,itr):
    kmeans = KMeans(n_clusters=elbow,n_init=1,init='k-means++',tol=toll,max_iter=itr)
    kmeans.fit(dataset)
    labels = kmeans.labels_
    centroidi = kmeans.cluster_centers_

    return labels, centroidi

def plotClusters(label,clusters):
    l = []
    sizes = []
    for i in range(clusters):
       sizes.append(0)
    for i in range(len(label)):
        sizes[label[i]] += 1

    for i in range(clusters):
        l.append(i)

    plt.pie(x=sizes,labels=l,autopct='%1.1f%%')
    plt.title('Clusters')
    plt.show()
    
dataset = pd.read_csv("./datasets/customer_segmentation_data.csv")
dataset = dataset.drop("id",axis=1)








#PRE-PROCESSING


datasetproc = pd.get_dummies(dataset,columns = ["gender","preferred_category"])

scaler = MinMaxScaler()
datasetproc = scaler.fit_transform(datasetproc)





#PCA
n_feats = datasetproc.shape[1]

print(f"Numero features prima della pca :{n_feats}")

elbowcomp,var,splits = getnumComp(datasetproc,n_feats)

pca_nfeats = int(np.round(elbowcomp*n_feats))

print(f"Numero features dopo la pca :{pca_nfeats}\nExplained Variance Ratio con {pca_nfeats} features usando la pca è : {np.round(var[int(elbowcomp*10) - 1],2)}")

pca = PCA(n_components=pca_nfeats)  
datasetproc = pd.DataFrame(pca.fit_transform(datasetproc))



#Clustering

param = {"tol": 1e-10,"itr":1000}


elbow,inertia = getnumClust(datasetproc,param["tol"],param["itr"])

labels,centroidi = clustering(datasetproc,elbow,param["tol"],param["itr"])













#save labels


dataset.insert(loc=len(dataset.columns),column="classe",value=labels)
dataset.to_csv("datasets/stacking/clusteredData.csv",index=False)






#plotting

plotKneeClust(inertia,range(1,100),elbow,'Inertia','Numero di Cluster k')

plotClusters(labels.tolist(),elbow)


plotKneeComp(var,splits,elbowcomp,'explained variance','features ratio')



print(f"SILOUHETTE SCORE: {silhouette_score(datasetproc,labels,metric="euclidean",random_state=42)}")

