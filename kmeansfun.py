import plot
import pandas as pd
import matplotlib.pyplot as plt 
from sys import maxsize
from sklearn.cluster import KMeans




def plotOneKmeansResult(dataset, nbClusters, minNumberOfClusterToPrintResults=-1, maxNumberOfClusterToPrintResults=maxsize, minClusterSize=0, maxClusterSize=maxsize, normalizedValues=False):
    # applying KMeans on the dataset
    model = KMeans(n_clusters=nbClusters)
    model.fit(dataset)

    # adding the cluster column to the dataset
    results = pd.DataFrame(model.labels_, columns=['cluster'])
    numberOfClusters = len(results.groupby('cluster').size())
    
    if numberOfClusters >= minNumberOfClusterToPrintResults and numberOfClusters <= maxNumberOfClusterToPrintResults:
        plot.plotHeatMapDistanceDepthValue(dataset=dataset,
                                                clustersDf=results, 
                                                minClusterSize=minClusterSize,
                                                maxClusterSize=maxClusterSize,
                                                normalizedValues=normalizedValues, 
                                                printPlottedClustersDf=False)
        
        plt.title(f'n_clusters={nbClusters}')
        plt.show()
    return results, model


