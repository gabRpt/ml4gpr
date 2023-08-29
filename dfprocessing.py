import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sys import maxsize




# Scaling headers numbers to nanoseconds
# eg: column 1 is 0 and column 510 is 30ns
def _createNanosecondsHeader(data):
    NB_MEASURES = data.shape[1] - 1
    MAX_NANOSECONDS = 30
    NANOSECONDS_DIVIDER = math.floor(NB_MEASURES / MAX_NANOSECONDS)

    column_names = ['distance']

    for i in range(0, NB_MEASURES):
        column_names.append(str(i / NANOSECONDS_DIVIDER))

    data.columns = column_names




def _cutDataframByTime(dataframe, startTimeInNanoseconds=0, endTimeInNanoseconds=maxsize):
    columnsNames = dataframe.columns.values[1:] # all excpect distance column
    columnsNames = columnsNames.astype(float)
    columnsNames = columnsNames[(columnsNames >= startTimeInNanoseconds) & (columnsNames <= endTimeInNanoseconds)]
    columnsNames = columnsNames.astype(str)
    columnsNames = np.append('distance', columnsNames)
    return dataframe.loc[:, columnsNames]


def createDataSample(data, startPointInMeters=1850, endPointInMeters=1960, startTimeInNanoseconds=0, endTimeInNanoseconds=maxsize,  generateOutputFile=True, outputFileName='sample.csv'):
    # Select the data between startPointInMeters and startPointInMeters
    # the first column defines the distance of the measure
    sample = data[(data['distance'] >= startPointInMeters) & (data['distance'] <= endPointInMeters)]
    _createNanosecondsHeader(sample)
    sample = _cutDataframByTime(sample, startTimeInNanoseconds, endTimeInNanoseconds)

    # save the sample as a csv file
    if generateOutputFile:
        sample.to_csv(outputFileName, sep=';', index=False)
        
    return sample





def getDataSample(fileName='sample.csv'):
    return pd.read_csv(fileName, sep=';')





# Reducing differences in the dataset
# iterate over the data (except the first column)
# and then power the values by 1/2
# if the value is negative we power it by 1/2 and keep it negative
def applyPowerToDataframe(dataframe, power=3/4):
    returnedDf = dataframe.copy()
    for i in range(1, len(returnedDf.columns)):
        col = returnedDf.columns[i]
        returnedDf[col] = np.power(returnedDf[col].abs(), power) * np.sign(returnedDf[col])
    return returnedDf





# Transpose the dataset (like matrix)
def transposeDataframe(dataframe):
    return dataframe.T




# Returning the same dataframe but with normalized data
def getNormalizedDataframe(dataframe, featureRange=(-1, 1), columnsToNormalize=None):
    xScaledDf = dataframe.copy()
    
    if columnsToNormalize is None:
        stdscaler = preprocessing.MinMaxScaler(feature_range=featureRange)
        xScaledDf = stdscaler.fit_transform(dataframe)
        xScaledDf = pd.DataFrame(xScaledDf)
        xScaledDf.columns = dataframe.columns
    else:
        for col in columnsToNormalize:
            xScaledDf[col] = preprocessing.MinMaxScaler(feature_range=featureRange).fit_transform(xScaledDf[[col]])        
        
    return xScaledDf





# Current structure of the dataset
# distance	0.0	0.058823529411764705	0.11764705882352941	
# 1850.00 0	0	-68
# 1850.05	0	0	-1
# 1850.10	0	0	-92

# Wanted structure of the dataset 
# distance    depth   value
# 1850.00 0.0 0
# 1850.00 0.058823529411764705 0
# 1850.00 0.11764705882352941 -68
# 1850.05 0.0 0
# 1850.05 0.058823529411764705 0
# 1850.05 0.11764705882352941 -1
# 1850.10 0.0 0
# 1850.10 0.058823529411764705 0
# 1850.10 0.11764705882352941 -92
def getDistanceDepthValueDataset(dataset):
    dataset = dataset.melt(id_vars=['distance'], value_vars=dataset.columns)
    dataset = dataset.rename(columns={"index": "distance", "variable": "depth", "value": "value"})
    dataset['depth'] = dataset['depth'].astype(float)
    dataset = dataset.sort_values(by=['distance','depth'])
    dataset = dataset.reset_index(drop=True)
    return dataset



def getModelMetrics(dataset, model):
    inetria = model.inertia_
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    # score = metrics.silhouette_score(dataset, model.labels_, metric='euclidean')
    # too much time to process
    
    return {'inertia':inetria}



def getClusterMetrics(dataset, clusters):
    workingDf = pd.concat([dataset, clusters], axis=1)
    
    for cluster in workingDf['cluster'].unique():
        clusterDf = workingDf[workingDf['cluster'] == cluster]
        print(f'Cluster {cluster}')
        print(clusterDf.describe())
        print('\n')



def getPointsMetrics(dataset, clusters):
    return

