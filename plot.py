import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

OUTLIERS_COLOR = 'black'



# for other colors check https://matplotlib.org/stable/gallery/color/colormap_reference.html
def getColorMap(dataStructure, colormap='rainbow'):
    arg = np.linspace(0, 1, len(dataStructure))
    palette = None
    if colormap == 'rainbow':
        palette = cm.rainbow(arg)
        
    elif colormap == 'tab20':
        palette = cm.tab20(arg)
        
    elif colormap == 'gist_rainbow':
        palette = cm.gist_rainbow(arg)
        
    else:
        raise Exception(f'colormap {colormap} is not supported')
    
    return palette





# returns the number of measures and the nanosecond divider
def getDatasetInformations(dataset, maximumNumberOfNanoseconds=30):
    nbMeasures = dataset.shape[1] - 1
    nanosecondsDivider = math.floor(nbMeasures / maximumNumberOfNanoseconds)
    return nbMeasures, nanosecondsDivider





# Plot the first row, showing x axis every 50 measure
def plotOneDatasetRow(dataset, row=1, maximumNumberOfNanoseconds=30):
    plt.plot(dataset.iloc[row, 1:], 'o')
    
    nbMeasures, nanosecondsDivider = getDatasetInformations(dataset, maximumNumberOfNanoseconds)

    # The columns headers range from 0 to 30
    # Change x ticks to show only header round values with 2 as a step : 0,2,4,..,26,28,30
    x_ticks = dataset.columns[1:]
    step = nanosecondsDivider * 2

    plt.xticks(np.arange(0, nbMeasures, step), x_ticks[::step].astype(float).astype(int))
    plt.xlabel('Time (ns)')
    plt.ylabel('Measured value')


    # Adding grid
    ax = plt.gca()
    ax.grid(color='grey', linestyle='--', linewidth=0.5)


    # Adding maximum and minimum measured value labels
    maxValue = dataset.iloc[row, 1:].max()
    minValue = dataset.iloc[row, 1:].min()
    maxValueColumnNumber = int(float(dataset.iloc[row, 1:].idxmax()) * nanosecondsDivider) + 5
    minValueColumnNumber = int(float(dataset.iloc[row, 1:].idxmin()) * nanosecondsDivider) + 5

    plt.text(maxValueColumnNumber, maxValue, 'max = ' + str(maxValue))
    plt.text(minValueColumnNumber, minValue, 'min = ' + str(minValue))





# Creates a plot where the color of each point is in a grey scale.
# highest value is black
# lowest value is white
# x axis is the distance
# y axis is the columns number
def generateHeatMap(dataset, cmap='gray', xTicksInterval=200, clustersDf=None, showClusters=False, maximumNumberOfNanoseconds=30, transposedDataset=False, colormap='rainbow'):
    if transposedDataset:
        dataset = dataset.T
    
    # dataset.iloc[:, 1:] dataset without column 'distance'
    plt.imshow(dataset.iloc[:, 1:].T, cmap=cmap, aspect='auto')

    nbMeasures, nanosecondsDivider = getDatasetInformations(dataset, maximumNumberOfNanoseconds)
    
    # Set the tick values on the x-axis
    x_ticks = np.arange(0, len(dataset), xTicksInterval)
    x_tick_labels = np.round(dataset.iloc[x_ticks, 0]).astype(int)
    plt.xticks(x_ticks, x_tick_labels)
    
    # calculate the number of minor ticks between each major ticks
    minorTicksInterval = xTicksInterval / 5

    # Setup x-axis
    ax = plt.gca() # select the current axis of the plot
    ax.xaxis.set_ticks_position('top') # Set the position of the x-axis ticks to the top
    ax.xaxis.set_label_position('top') # Set the position of the x-axis label to the top
    ax.xaxis.set_minor_locator(plt.MultipleLocator(minorTicksInterval)) # Add small ticks between each big ticks
    plt.xlabel('Distance (m)')

    # Setup y-axis
    # Set the tick values on the y-axis
    y_ticks = dataset.columns[1:]
    step = nanosecondsDivider * 5
    plt.yticks(np.arange(0, nbMeasures, step), y_ticks[::step].astype(float).astype(int))
    plt.ylabel('Time (ns)')

    # Add horizontal lines for each y-axis tick (except the first one)
    y_ticks = ax.get_yticks()
    for y in y_ticks[1:]:
        ax.axhline(y=y, color='black', linestyle='--', linewidth=0.8)

    # Add another y axis on the other side of the plot
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_ticks)
    ax2.set_ylabel('Depth measure number')
    
    # Adding cluster visualization    
    if clustersDf is not None and showClusters:
        clusters = clustersDf['cluster'].unique()
        colors = getColorMap(clusters, 'gist_rainbow')
        
        for cluster, c in zip(clustersDf.index.values, colors):
            linesInCurrentCluster = clustersDf[clustersDf['cluster'] == cluster]
            
            for line in linesInCurrentCluster.index.values:
                if transposedDataset:
                    ax.axhline(y=line, color=c, linestyle='-', linewidth=1.5)
                    ax.lines[-1].set_alpha(0.2)
                else:
                    ax.axvline(x=line, color=c, linestyle='-', linewidth=1.5)
                    ax.lines[-1].set_alpha(0.1)
        




def plotHeatMapDistanceDepthValue(dataset,
                                  clustersDf=None, 
                                  minClusterSize=0,
                                  maxClusterSize=sys.maxsize,
                                  ignoredClusters=[],
                                  clustersToShow=[],
                                  normalizedValues=False, 
                                  printPlottedClustersDf=False,
                                  alpha=1.0,
                                  overlayOriginalPlot=False,
                                  colorPalette='tab20'):
    
    if overlayOriginalPlot:
        plotHeatMapDistanceDepthValue(dataset)
    
    if normalizedValues:
        minDistance = dataset['distance'].min()
        maxDistance = dataset['distance'].max()
        minDepth = dataset['depth'].min()
        maxDepth = dataset['depth'].max()
        
        plt.ylim(minDepth, maxDepth)
        plt.xlim(minDistance, maxDistance)
    
    if clustersDf is None:
        plt.scatter(dataset['distance'], dataset['depth'], c=dataset['value'], cmap='gray', alpha=alpha)
    else:        
        datasetCopy = pd.concat([dataset, clustersDf], axis=1)
        clustersArray = datasetCopy.groupby('cluster').size()
        clustersArray = clustersArray[clustersArray > minClusterSize]
        clustersArray = clustersArray[clustersArray < maxClusterSize]
        clustersArray = clustersArray.sort_values(ascending=False)
        
        # removing ignored clusters
        clustersArray = clustersArray.drop(ignoredClusters, errors='ignore')
        
        # keep only clusters to show
        if len(clustersToShow) > 0 and clustersArray is not None:
            clustersArray = clustersArray[clustersArray.index.isin(clustersToShow)]
        
        
        if printPlottedClustersDf:
            print(clustersArray)
        
        colors = getColorMap(clustersArray, colorPalette)
        legend_labels = []
        
        for cluster, c in zip(clustersArray.index.values, colors):
            plt.scatter(datasetCopy['distance'][datasetCopy['cluster'].isin([cluster])], datasetCopy['depth'][datasetCopy['cluster'].isin([cluster])], color=c, s=0.1, alpha=alpha)
            legend_labels.append('Cluster ' + str(cluster))

        lgnd = plt.legend(legend_labels, title='Clusters', bbox_to_anchor=(1, 1), loc='upper left', fontsize=14)
        for handle in lgnd.legendHandles:
            handle.set_sizes([50])

    if not overlayOriginalPlot:
        plt.gca().invert_yaxis()

