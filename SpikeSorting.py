import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def computeThreshold(data,factor):
    noise = data[0:500]
    noise = np.array(noise)
    return (factor * noise.std())

def rectifySignal(signal):
    for i in range(len(signal)):
        signal[i] = abs(signal[i])
    return signal


def calculateNoise(signal,threshold):
    noise = []
    for i in range(signal.shape[0]):
        if(signal[i] <= threshold):
            noise.append(signal[i])
    return noise

def get2dRepresentation(spikes,electrode_number,factor,saveFig):
    pca = PCA(n_components = 2)
    # np.reshape(spikes,(49,1))
    # print(spikes.shape)
    spikes_pca = pca.fit_transform(spikes)
    x=[]
    y=[]
    # print(spikes_pca.shape[0])
    # print(spikes_pca[0])
    for i in range(spikes_pca.shape[0]):
        x.append(spikes_pca[i][0])
        y.append(spikes_pca[i][1])
    fig1 = plt.figure(figsize=(20,10))
    plt.scatter(x,y)
    if(saveFig):
        if(factor == 3.5):
            plt.savefig("FeatureSpace_{}_3_5.jpg".format(electrode_number))
        elif factor ==5:
            plt.savefig("FeatureSpace_{}_5.jpg".format(electrode_number))
    return spikes_pca

def detectSpikePeaks(filtered_signal,tWindow,threshold):
    spikes = []
    offsets =[]
    i=tWindow//2
    while i < len(filtered_signal)-tWindow//2:
        # print(np.average(filtered_signal[i:i+tWindow]))
        if(filtered_signal[i]>threshold):
            index = locatePeak(filtered_signal[i-(tWindow//2):i+(tWindow//2)+1],threshold,i-tWindow//2)
            if(index>0):            
                spikes.append(filtered_signal[index-(tWindow//2):index+(tWindow//2)+1])
                offsets.append(index)
                i += tWindow//2 
            else:
                i+=1
        else:
            i+=1
    spikes=np.array(spikes)
    # print(spikes.shape)
    return {"spikes":spikes,"offsets":offsets}

def detectClusters(spikes,electrode_number,factor,saveFig):
    number_of_clusters = 3
    if(electrode_number==2 or factor == 5):
        number_of_clusters = 2
    kmeans = KMeans(number_of_clusters)
    kmeans.fit(spikes)
    labels = kmeans.predict(spikes)
    centroids = kmeans.cluster_centers_
    fig1 = plt.figure(figsize=(20,10))
    plt.scatter(spikes[:,0],spikes[:,1],c=labels,cmap='rainbow')
    plt.scatter(centroids[:,0],centroids[:,1],color='black',marker='x',s=150)
    if(saveFig):
        if(factor == 3.5):
            plt.savefig("Clusters_{}_3_5.jpg".format(electrode_number))
        elif factor ==5:
            plt.savefig("Clusters_{}_5.jpg".format(electrode_number))
    plt.close()
    return {"labels":labels,"centroids":centroids}




# def detectSpikePeaks(rectifiedSignal,tWindow,threshold):
#     spikes = []
#     offsets =[]
#     i = tWindow//2
#     noise = np.array(calculateNoise(rectifiedSignal,threshold))
#     threshold = np.std(noise) * 3
#     while i < len(rectifiedSignal)-tWindow:
#         if(np.average(rectifiedSignal[i:i+tWindow])>threshold):
#             index = locatePeak(rectifiedSignal[i-tWindow//2:i+(tWindow//2)+1],threshold,i-tWindow//2)
#             x = rectifiedSignal[index-(tWindow//2):index+(tWindow//2)+1]
#             if(x.shape[0] == tWindow and index>0):
#                 spikes.append(x)
#                 offsets.append(index)
#                 i+= tWindow//2
#             else:
#                 i+=1
#         else:
#             i+=1
#     spikes=np.array(spikes)
#     print(spikes.shape)
#     return {"spikes":spikes,"offsets":offsets}

def locatePeak(data,threshold,offset):
    max = data[0]
    index_max = 0
    for i in range(len(data)):
        if(data[i]>max):
            max=data[i]
            index_max=i
    # if(max> threshold):
    return index_max+offset
    # else:
    #     return -1

def plotDetectedSpikes(rectified_data,spikesDict,electrode_number,labels,factor):
    fig1 = plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(rectified_data)),rectified_data)
    spikesPeaks = []
    spikes = spikesDict["spikes"]
    offsets = spikesDict["offsets"]
    for i in range(len(offsets)):
        spikesPeaks.append(rectified_data[offsets[i]])
    plt.scatter(offsets,spikesPeaks,c=labels,cmap='rainbow')
    if(factor == 3.5):
        plt.savefig("DetectedSpikes{}_3_5.jpg".format(electrode_number))
    elif factor ==5:
        plt.savefig("DetectedSpikes{}_5.jpg".format(electrode_number))
    plt.close()

def plotSpikesMeans(spikesToElectrode,labestoElectrode,factor):
    fig = plt.figure(figsize=(20,20))
    ourColors={"1":"red","2":"black","3":"green"}
    counter=1
    for key in labestoElectrode.keys():
        uniqueLabels = np.unique(labestoElectrode[key])
        for i in range(len(uniqueLabels)):
            if(factor==3.5):
                ax1 = fig.add_subplot(int("23{}".format(counter)))
            elif factor ==5:
                ax1 = fig.add_subplot(int("22{}".format(counter)))
            counter+=1
            # spikes = [spike for spike,index in spikesToElectrode[key] if ]
            spikes = [spike for spike,label in zip(spikesToElectrode[key],labestoElectrode[key]) if label == uniqueLabels[i]]
            spikes_mean = np.mean(spikes,axis=0)
            ourColor = "blue"
            if(ourColors.__contains__(str(uniqueLabels[i]+1))):
                ourColor = ourColors[str(uniqueLabels[i]+1)]
            ax1.plot(np.arange(len(spikes_mean)),spikes_mean,color=ourColor)
    if(factor==3.5):
        plt.savefig("Templates_3_5.jpg")
    elif factor==5:
        plt.savefig("Templates_5.jpg")
    plt.close()
        
    # for key in returnData['detectedMus'].keys():
    #         ax1 = fig.add_subplot(int("31{}".format(key)))
    #         muap = returnData['detectedMus'][key]
    #         t = np.arange(len(muap))
    #         # print(muap)
    #         # print(t)
    #         ourColor = "blue"
    #         if(ourColors.__contains__(key)):
    #             ourColor = ourColors[key]
    #         ax1.plot(t,muap,color=ourColor)
    # plt.savefig("Templates.jpg")


def startSpikeVisualization():
    data = np.loadtxt("Data.txt")
    tWindow=49
    data= np.transpose(data)
    factors = [3.5,5]
    spkiesToElectrode = {}
    labelsToElectrode = {}
    for j in range (len(factors)):
        for i in range(data.shape[0]):
            our_threshold = computeThreshold(data[i],factor=factors[j])
            spikesDict = detectSpikePeaks(data[i],tWindow,our_threshold)
            spikes_pca = get2dRepresentation(spikesDict["spikes"],i+1,factors[j],True)
            labelsDict = detectClusters(spikes_pca,i+1,factors[j],True)
            spkiesToElectrode[i+1] = spikesDict["spikes"]
            labelsToElectrode[i+1] = labelsDict["labels"]
        plotSpikesMeans(spkiesToElectrode,labelsToElectrode,factors[j])
        

def startPlottingSpikesOnData(limit):
    data = np.loadtxt("Data.txt")
    tWindow=49
    data= np.transpose(data[0:limit])
    factors = [3.5,5]
    for i in range(data.shape[0]):
        for j in range (len(factors)):
            our_threshold = computeThreshold(data[i],factor=factors[j])
            spikesDict = detectSpikePeaks(data[i],tWindow,our_threshold)
            spikes_pca = get2dRepresentation(spikesDict["spikes"],i+1,factors[j],False)
            lablesDict = detectClusters(spikes_pca,i+1,factors[j],False)
            plotDetectedSpikes(data[i],spikesDict,i+1,lablesDict["labels"],factors[j])

if __name__ == "__main__":
    startSpikeVisualization()
    startPlottingSpikesOnData(20000)

   
# x=[[3,4],[5,2]]
# # y=x.index(3)
# print(np.mean(x,axis=0))
