import plot
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sys import maxsize
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors




def getDictionnaryForMultipleDBSCANRepresentations(epsMin, epsMax, epsStep, minSamplesMin, minSamplesMax, minSamplesStep):
    returnedDictionnary = {}
    for eps in np.arange(epsMin, epsMax, epsStep):
        returnedDictionnary[eps] = []
        for min_samples in np.arange(minSamplesMin, minSamplesMax, minSamplesStep):
            returnedDictionnary[eps].append(min_samples)
    return returnedDictionnary





def plotMultipleDBSCANRepresentations(dataset, epsAndSampleToTest, minNumberOfClusterToPrintResults=-1, maxNumberOfClusterToPrintResults=maxsize, minClusterSize=0, maxClusterSize=maxsize, transposedDataset=False, distanceDepthValueDf=False, normalizedValues=False):
    for eps in epsAndSampleToTest:
        for min_samples in epsAndSampleToTest[eps]:
            plotOneDBSCANResult(dataset, 
                                eps, 
                                min_samples, 
                                minNumberOfClusterToPrintResults=minNumberOfClusterToPrintResults, 
                                maxNumberOfClusterToPrintResults=maxNumberOfClusterToPrintResults,
                                minClusterSize=minClusterSize,
                                maxClusterSize=maxClusterSize,
                                transposedDataset=transposedDataset, 
                                distanceDepthValueDf=distanceDepthValueDf,
                                normalizedValues=normalizedValues)
            plt.show()





# eps : maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples : The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
def plotOneDBSCANResult(dataset, eps, min_samples, minNumberOfClusterToPrintResults=-1, maxNumberOfClusterToPrintResults=maxsize, minClusterSize=0, maxClusterSize=maxsize,  transposedDataset=False, distanceDepthValueDf=False, normalizedValues=False, alpha=1.0):
    # Create DBSCAN object
    model = DBSCAN(eps=float(eps), min_samples=min_samples)
    model.fit(dataset)

    results = pd.DataFrame(model.labels_, columns=['cluster'])
    numberOfClusters = len(results.groupby('cluster').size())
    
    if numberOfClusters >= minNumberOfClusterToPrintResults and numberOfClusters <= maxNumberOfClusterToPrintResults:
        if distanceDepthValueDf:
            plot.plotHeatMapDistanceDepthValue(dataset=dataset,
                                                clustersDf=results, 
                                                minClusterSize=minClusterSize,
                                                maxClusterSize=maxClusterSize,
                                                normalizedValues=normalizedValues,
                                                alpha=alpha,
                                                printPlottedClustersDf=False)
        else:
            print(results.groupby('cluster').size())
            plot.generateHeatMap(dataset, clustersDf=results, showClusters=True, transposedDataset=transposedDataset)
        
        plt.title(f'eps={eps}, min_sample={min_samples}')
            
    
    return results





def plotNearestNeighbours(dataset, n_neighbors, ylimMin=None, ylimMax=None):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    
    if ylimMax is not None and ylimMin is not None:
        plt.ylim(ylimMin, ylimMax)
    
    plt.plot(distances)
    plt.show()