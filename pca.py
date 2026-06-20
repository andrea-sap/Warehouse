from sklearn.decomposition import PCA
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt



def getnumComp(dataSet,n_feats):

    TotexVarRatio = []
    splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9]

    for split in splits:
        featsplit =int(np.round(split*n_feats))
        pca = PCA(n_components=featsplit)
        pca.fit(dataSet)
        totratio = 0
        var = pca.explained_variance_ratio_
        for i in var:
            totratio = totratio + i
        TotexVarRatio.append(totratio)
    kl = KneeLocator(splits, TotexVarRatio, curve="concave", direction="increasing")
    
    
    return float(kl.elbow),TotexVarRatio,splits  


def plotKneeComp(y,x,elbow,ylabel,xlabel):
    plt.plot(x, y, 'bx-')
    e = int(elbow*10) - 1
    plt.scatter(elbow, y[e], c='red', label=f'best k: {elbow}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Ricerca k ottimale')
    plt.legend()
    plt.show()