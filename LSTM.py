import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LSTMClass import LSTMClass
import sys
import os
import sklearn.preprocessing
import pandas as pd
import statistics 
from matplotlib.offsetbox import AnchoredText

# splitting the data in 80-20% train-test sets
testSizePercentage = 20 

# read the dataset
df = pd.read_csv("prices-split-adjusted.csv", index_col = 0)

# number of different stocks
print('\n Total Number of stocks: ', len(list(set(df.symbol))))
print('Some of the stock symbols: ', list(set(df.symbol))[:10])

stockName = sys.argv[1]#'MSFT'   ## selct on which stock you want to predict
plt.figure(figsize=(20, 8));
plt.subplot(1,2,1);
plt.plot(df[df.symbol == stockName].open.values, color='green', label='open')    
plt.plot(df[df.symbol == stockName].close.values, color='black', label='close')
plt.plot(df[df.symbol == stockName].low.values, color='red', label='low')
plt.plot(df[df.symbol == stockName].high.values, color='blue', label='high')
plt.title('Stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
plt.subplot(1,2,2);
plt.plot(df[df.symbol == stockName].volume.values, color='red', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best');
subset = df[df.symbol == stockName].loc[:,list(df.keys())[1:]]
minMaxScaler = sklearn.preprocessing.MinMaxScaler()

def preProcessData(df):
    df['open'] = minMaxScaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = minMaxScaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = minMaxScaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = minMaxScaler.fit_transform(df['close'].values.reshape(-1,1))
    return df
dfStock = df[df.symbol == stockName].copy()
dfStock.drop(['symbol'],1,inplace=True)
dfStock.drop(['volume'],1,inplace=True)

columnName = list(dfStock.columns.values)
print('dfStock.columns.values = ', columnName)

# pre-process stock
dfStockPreProcess = dfStock.copy()
dfStockPreProcess = preProcessData(dfStockPreProcess)
dfStockPreProcess.describe()
print(dfStockPreProcess[:5])
def loadData(stock, sequenceLength):
    rawData = stock.as_matrix() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(rawData) - sequenceLength): 
        data.append(rawData[index: index + sequenceLength])
    
    data = np.array(data);
    testSetSize = int(np.round(testSizePercentage/100*data.shape[0]));
    trainSetSize = data.shape[0] - (testSetSize);
    
    x_train = data[:trainSetSize,:-1,:]
    y_train = data[:trainSetSize,-1,:]
    x_test = data[trainSetSize:,:-1,:]
    y_test = data[trainSetSize:,-1,:]
    
    return [x_train, y_train, x_test, y_test]
# create train, test data
sequenceLength = 19 # choose sequence length
x_train, y_train, x_test, y_test = loadData(dfStockPreProcess, sequenceLength+1)
# Choose only open prices
x_train, y_train, x_test, y_test = x_train[:,:,0], y_train[:,0], x_test[:,:,0], y_test[:,0]

sequenceLength = 19 
inputSize = 1
hiddenLayerSize = int(sys.argv[2])
outputSize = 1
learningRate = float(sys.argv[3])
n, p = 0, 0

outputWeight = np.random.randn(outputSize, hiddenLayerSize) / np.sqrt(outputSize)
lstmObj = LSTMClass(inputSize, hiddenLayerSize)
input = np.zeros((sequenceLength,1))
target = np.zeros((sequenceLength,outputSize))
modifiedOutputWeight = np.zeros_like(outputWeight)

j=0
k=0
trainingActualValues = []
trainingPredictedValues = []
trainingLoss = []
for i in range(1394):
    if j + sequenceLength + outputSize >= len(x_train):
        j=0
        lstmObj.resetStates()
    input[:,0] = x_train[j,:]
    target[:,0] = y_train[j]
    trainingActualValues.append(np.mean(np.square(target)))
    lstmObj.forward(input)  
    hiddenLayerOutput = lstmObj.getHiddenOutput()  
    output = hiddenLayerOutput.dot(outputWeight.T) 
    trainingPredictedValues.append(np.mean(output))
    error = output - target
    modifiedWeightsForBackWardPass = (error).T.dot(hiddenLayerOutput)  
    loss = np.mean(np.square(output - target))  
    trainingLoss.append(loss)
    outputForBackWardPass = (error).dot(outputWeight)
    lstmObj.backward(outputForBackWardPass)
    lstmObj.trainNetwork(learningRate)        
    for param, dparam, mem in zip([outputWeight],
                              [modifiedWeightsForBackWardPass],
                              [modifiedOutputWeight]):
        mem += dparam * dparam
        param += -learningRate * dparam / np.sqrt(mem + 1e-8)
    
    print (k, loss)

    k += 1
    j += 1

print("Training Loss: ", str(np.sum(trainingLoss)))
print("Training Accuracy: ", 100 - (abs(statistics.mean([trainingPredictedValues_i - trainingActualValues_i for trainingPredictedValues_i, trainingActualValues_i in zip(trainingPredictedValues, trainingActualValues)])))*100 )
##################PREDICTING PHASE##########################

testingActualValues = []
testingPredictedValues = []
testingLoss = []
j = 0
for i in range(348):
    if j + sequenceLength + outputSize >= len(x_test):
        j=0
        lstmObj.resetStates()
    input[:,0] = x_test[j,:]
    target[:,0] = y_test[j]
    testingActualValues.append(np.mean(np.square(target)))
    lstmObj.forward(input)  
    hiddenLayerOutput = lstmObj.getHiddenOutput()  
    output = hiddenLayerOutput.dot(outputWeight.T)   
    testingPredictedValues.append(np.mean(output))
    loss = np.mean(np.square(output - target))  
    testingLoss.append(loss)
    j += 1

print("Testing Loss: ", str(np.sum(testingLoss)))
print("Testing Accuracy: ", 100 - (abs(statistics.mean([testingPredictedValues_i - testingActualValues_i for testingPredictedValues_i, testingActualValues_i in zip(testingPredictedValues, testingActualValues)])))*100 )
plt.figure(3)
plt.plot(testingActualValues, color='red', label='actual')
plt.plot(testingPredictedValues, color='blue', label='predicted')
#plt.yticks(np.arange(0, max(predictedValues)+0.01, 0.01))
plt.title('Prediction vs Actual Values of Stocks')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best');
#plt.text(0.02, 0.5, "Loss: "+str(np.sum(totalLoss)), fontsize=14, transform=plt.gcf().transFigure)
text_box = AnchoredText("Loss: "+str(round(np.sum(testingLoss), 2)) + '\n' + "Accuracy: " + str(round(100 - (abs(statistics.mean([testingPredictedValues_i - testingActualValues_i for testingPredictedValues_i, testingActualValues_i in zip(testingPredictedValues, testingActualValues)])))*100, 2)) , frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.show()
