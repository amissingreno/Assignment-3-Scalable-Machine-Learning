# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:36:53 2024

@author: Robert Martin
"""

import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#Function to resize image
def resizeImage(image):
    height, width = image.shape
    width = round((width*(256/height)) / 16) * 16
    height = 256
    imageResized = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    imageResized = cv2.normalize(imageResized.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
    #print(imageResized.shape)
    return imageResized

#Function to create 16x16 block vectors for images
def createBlockFeatureVectors(imageG, label, fileSuffix):
    nn = 16
    height, width = imageG.shape    
    data = []
   
    for ii in range(0, height, nn):
        for jj in range(0, width, nn):
            block = imageG[ii:ii+16, jj:jj+16]
            flattened = block.flatten()
            data.append(flattened)

    data = np.vstack(data)

    df1 = pd.DataFrame(data)
    
    if (label == 'cardinal'):
        df1[len(flattened) + 1 ] = 0
    elif (label == 'sparrow'):
        df1[len(flattened) + 1 ] = 1
    else:
        df1[len(flattened) + 1 ] = 2

    df1.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out' + fileSuffix + '.csv', index=False, header=False)
    return df1

#Function to create sliding block vectors for images
def createSlidingFeatureVectors(imageG, label, fileSuffix):
    height, width = imageG.shape
    data = []
   
    for ii in range(0, height-16):
        for jj in range(0, width-16):
            block = imageG[ii:ii+16,jj:jj+16]
            flattened= block.flatten()
            data.append(flattened)

    data = np.vstack(data)

    df1 = pd.DataFrame(data)
    
    if (label == 'cardinal'):
        df1[len(flattened) + 1] = 0
    elif (label == 'sparrow'):
        df1[len(flattened) + 1] = 1
    else:
        df1[len(flattened) + 1] = 2
        
    df1.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out' + fileSuffix + '.csv', index=False, header=False)
    return df1

#Function to open file and add contents to a vector
def openFile(fileName):
    file = open(fileName)
    reader = pd.read_csv(fileName)
    file.close()
    return reader

#Function to count number of observations
def observationCount(dataSet):
    rows, columns = dataSet.shape
    return rows

#Function to count number of features
def featureCount(dataSet):
    rows, columns = dataSet.shape
    return columns - 1
  
#Function to determine high dimensionality  
def dimensionality(dataSet):
    features = featureCount(dataSet)
    observations = observationCount(dataSet)
    
    if (features > observations):
        return True
    else:
        return False

#Function to calculate and create list of mean values for each feature
def featureMeanDataset(dataSet):
    meanList = []
    
    for ii in range(0, len(dataSet.columns) - 1):
        meanList.append(np.mean(dataSet.iloc[:,ii]))
        
    return meanList

#Function to calculate and create list of variance for each feature
def featureVarianceDataset(dataSet):
    varianceList = []
    
    for ii in range(0, len(dataSet.columns) - 1):
        varianceList.append(np.var(dataSet.iloc[:, ii]))
    
    return varianceList

#Function to calculate and create list of standard deviation for each feature
def featureStandDevDataset(dataSet):
    stdList = []
    
    for ii in range(0, len(dataSet.columns) - 1):
        stdList.append(np.std(dataSet.iloc[:, ii]))   
        
    return stdList

#Function to merge two feature vectors and shuffle rows
def mergeTwoFeatureVectors16(featureVector1, featureVector2, fileSuffix):
    completeVector = []
    
    completeVector.append(featureVector1)
    completeVector.append(featureVector2)

    completeVector = np.vstack(completeVector)
    
    fs1 = pd.DataFrame(completeVector)
    fs1 = fs1.sample(frac=1).reset_index(drop=True)
    
    fs1.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image01-' + fileSuffix + '.csv', index=False, header=False)
    return completeVector

#Function to merge three feature vectors and shuffle rows
def mergeThreeFeatureVectors16(featureVector1, featureVector2, featureVector3, fileSuffix):
    completeVector = []
    
    completeVector.append(featureVector1)
    completeVector.append(featureVector2)
    completeVector.append(featureVector3)
    
    completeVector = np.vstack(completeVector)
    
    fs2 = pd.DataFrame(completeVector)
    fs2 = fs2.sample(frac=1).reset_index(drop=True)
    
    fs2.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image012-' + fileSuffix + '.csv', index=False, header=False)
    return completeVector
    #return fs2.values.tolist()
    
#Function to merge two feature vectors and shuffle rows
def mergeTwoFeatureVectorsSliding(featureVector1, featureVector2, fileSuffix):
    completeVector = []
    
    completeVector.append(featureVector1)
    completeVector.append(featureVector2)

    completeVector = np.vstack(completeVector)
    
    fs1 = pd.DataFrame(completeVector)
    fs1 = fs1.sample(frac=1).reset_index(drop=True)
    
    fs1.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image01-' + fileSuffix + '.csv', index=False, header=False)
    return completeVector 

#Function to merge three feature vectors and shuffle rows
def mergeThreeFeatureVectorsSliding(featureVector1, featureVector2, featureVector3, fileSuffix):
    completeVector = []
    
    completeVector.append(featureVector1)
    completeVector.append(featureVector2)
    completeVector.append(featureVector3)
    
    completeVector = np.vstack(completeVector)
    
    fs2 = pd.DataFrame(completeVector)
    fs2 = fs2.sample(frac=1).reset_index(drop=True)
    
    fs2.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image012-' + fileSuffix + '.csv', index=False, header=False)
    return completeVector
    #return fs2.values.tolist()

def split8020dataset(NN, mergedDataset, fileNameX, fileNameY):

    # Label/Response set
    y = mergedDataset[NN]

    # Drop the labels and store the features
    mergedDataset.drop(NN,axis=1,inplace=True)
    X = mergedDataset

    # Generate feature matrix using a Numpy array
    tmp = np.array(X)
    X1 = tmp[:,0:NN]

    # Generate label matrix using Numpy array
    Y1 = np.array(y)

    # Split the data into 80:20
    row, col = X.shape

    TR = round(row*0.8)
    TT = row-TR

    # Training with 80% data
    X1_train = X1[0:TR-1,:]
    Y1_train = Y1[0:TR-1]
    
    #Save the 80% train datasets
    convertToDataFrameAndSaveTrain(X1_train, fileNameX)
    convertToDataFrameAndSaveTrain(Y1_train, fileNameY)
    
    #Create 20% data subset 
    X1_test = X1[TR:row,:]
    Y1_test = Y1[TR:row]
    
    #Save the 20% test datasets
    convertToDataFrameAndSaveTest(X1_test, fileNameX)
    convertToDataFrameAndSaveTest(Y1_test, fileNameY)
    
    return X1_train, Y1_train, X1_test, Y1_test
    
def convertToDataFrameAndSaveTrain(array, fileID):
    df = pd.DataFrame(array)
    df.to_csv('C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/' + fileID + '_train.csv', index=False, header=False)
    
def convertToDataFrameAndSaveTest(array, fileID):
    df = pd.DataFrame(array)
    df.to_csv('C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/' + fileID + '_test.csv', index=False, header=False)

def saveConfusionMatrix(confusion_matrix, fileID):
    df = pd.DataFrame(confusion_matrix)
    df.to_csv('C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/' + fileID + '_confusion_matrix.csv', index=False, header=False)
    
def saveActualAndPredictedTest(actual, predicted, fileID):
    temp = np.vstack(actual)
    yhat_test = np.vstack(np.round(abs(predicted)))
    predict = np.hstack((temp, yhat_test))

    df = pd.DataFrame(predict)
    df.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/' + fileID + '_y_and_yhat.csv', index=False, header=False)

# Use randomized search function to determine the best number of trees in forest and depth of the tree
def randomForestBestParameters(X_train, Y_train, X_test):
    parameter_distributions = {'n_estimators': range(100,500),'max_depth': range(1,25)}
    rf = RandomForestClassifier()
    random_search = RandomizedSearchCV(rf, param_distributions=parameter_distributions)

    random_search.fit(X_train, Y_train)
    print('Best estimator:', random_search.best_params_)
    print()
    rf2 = RandomForestClassifier(n_estimators=random_search.best_params_.get('n_estimators'), max_depth=random_search.best_params_.get('max_depth'))
    rf2.fit(X_train, Y_train)

    yhat_test = rf2.predict(X_test)
    
    return yhat_test

def standardize(NN, dataSet, fileSuffix):   
    
    # Label/Response set
    y = dataSet[NN]
    # Convert to an array
    labels = np.array(y)
    
    # Drop the labels and store the features
    dataSet.drop(NN,axis=1,inplace=True)
    X = dataSet
    
    # Generate feature matrix using a Numpy array
    X1 = np.array(X)

    # Standardize the features.
    scaler = StandardScaler()
    x_std = scaler.fit_transform(X1)

    # Combine standardized feature space with labels to save as file.
    completeMatrix = np.column_stack([x_std, labels])
   
    std1 = pd.DataFrame(completeMatrix)
    std1.to_csv('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/standardized' + fileSuffix + '.csv', index=False, header=False)
    
    return std1

'''
# Open and display original bird images   
cardinal = cv2.imread("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Images/image0.jpg")
sparrow = cv2.imread("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Images/image1.jpg")
robin = cv2.imread("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Images/image2.jpg")

cardinal = cv2.cvtColor(cardinal, cv2.COLOR_BGR2RGB)
sparrow = cv2.cvtColor(sparrow, cv2.COLOR_BGR2RGB)
robin = cv2.cvtColor(robin, cv2.COLOR_BGR2RGB)

plt.imshow(cardinal)
plt.axis('off')
plt.show()

plt.imshow(sparrow)
plt.axis('off')
plt.show()

plt.imshow(robin)
plt.axis('off')
plt.show()

# Display the color channels - Cardinal
plt.imshow(cardinal[:,:,0])
plt.axis('off')
plt.show()

plt.imshow(cardinal[:,:,1])
plt.axis('off')
plt.show()

plt.imshow(cardinal[:,:,2])
plt.axis('off')
plt.show()

plt.imshow(cardinal[:,:,2],'gray')
plt.axis('off')
plt.show()

#Display the color channels - Sparrow
plt.imshow(sparrow[:,:,0])
plt.axis('off')
plt.show()

plt.imshow(sparrow[:,:,1])
plt.axis('off')
plt.show()

plt.imshow(sparrow[:,:,2])
plt.axis('off')
plt.show()

plt.imshow(sparrow[:,:,2],'gray')
plt.axis('off')
plt.show()

#Display the color channels - Robin
plt.imshow(robin[:,:,0])
plt.axis('off')
plt.show()

plt.imshow(robin[:,:,1])
plt.axis('off')
plt.show()

plt.imshow(robin[:,:,2])
plt.axis('off')
plt.show()

plt.imshow(robin[:,:,2],'gray')
plt.axis('off')
plt.show()

# Convert to grayscale and display image dimensions
cardinalG = cv2.cvtColor(cardinal, cv2.COLOR_BGR2GRAY) 
sparrowG = cv2.cvtColor(sparrow, cv2.COLOR_BGR2GRAY) 
robinG = cv2.cvtColor(robin, cv2.COLOR_BGR2GRAY)

plt.imshow(cardinalG, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()
print("Cardinal grayscale image dimensions: ", cardinalG.shape)

plt.imshow(sparrowG, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()
print("Sparrow grayscale image dimensions: ", sparrowG.shape)

plt.imshow(robinG, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()
print("Robin grayscale image dimensions: ", robinG.shape, "\n")

# Raw data (image) resizing
cardinalResized = resizeImage(cardinalG)
print("The resized cardinal grayscale image dimensions are: ", cardinalResized.shape)
sparrowResized = resizeImage(sparrowG)
print("The resized sparrow grayscale image dimensions are: ", sparrowResized.shape)
robinResized = resizeImage(robinG)
print("The resized robin grayscale image dimensions are: ", robinResized.shape, "\n")

# Plot the images
plt.imshow(cardinalResized, cmap=plt.get_cmap('gray'))
#plt.axis('off')
plt.show()

plt.imshow(sparrowResized, cmap=plt.get_cmap('gray'))
#plt.axix('off')
plt.show()

plt.imshow(robinResized, cmap=plt.get_cmap('gray'))
#plt.axis('off')
plt.show()

############################################################## CARDINAL 16X16

mm1 = []
for ii in range(0, 256, 16):
    for jj in range(0, 144, 16):
        blk1 = cardinalResized[ii:ii+16,jj:jj+16]
        flt1 = blk1.flatten()
        mm1.append(flt1)

mm1 = np.vstack(mm1)

df1 = pd.DataFrame(mm1)
df1[len(flt1)] = 0   # label 0 for cardinal

df1.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out1.csv", index=False, header=False)

############################################################## SPARROW 16X16

mm2 = []
for ii in range(0, 256, 16):
    for jj in range(0, 144, 16):
        blk2 = sparrowResized[ii:ii+16,jj:jj+16]
        flt2 = blk2.flatten()
        mm2.append(flt2)

mm2 = np.vstack(mm2)

df2 = pd.DataFrame(mm2)
df2[len(flt2)] = 1   # label 1 for sparrow

df2.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out2.csv", index=False, header=False)

############################################################## ROBIN 16X16

mm3 = []
for ii in range(0, 256, 16):
    for jj in range(0, 336, 16):
        blk3 = robinResized[ii:ii+16,jj:jj+16]
        flt3 = blk3.flatten()
        mm3.append(flt3)

mm3 = np.vstack(mm3)

df3 = pd.DataFrame(mm3)
df3[len(flt3)] = 2   # label 2 for robin

df3.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out3.csv", index=False, header=False)

############################################################## CARDINAL

mm4 = []
for ii in range(0, 240):
    for jj in range(0, 128):
        blk4 = cardinalResized[ii:ii+16,jj:jj+16]
        flt4 = blk4.flatten()
        mm4.append(flt4)

mm4 = np.vstack(mm4)

df4 = pd.DataFrame(mm4)
df4[len(flt4)] = 0   # label 0 for cardinal

df4.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out4.csv", index=False, header=False)

############################################################## SPARROW

mm5 = []
for ii in range(0, 240):
    for jj in range(0, 128):
        blk5 = sparrowResized[ii:ii+16,jj:jj+16]
        flt5 = blk5.flatten()
        mm5.append(flt5)

mm5 = np.vstack(mm5)

df5 = pd.DataFrame(mm5)
df5[len(flt5)] = 1   # label 1 for sparrow

df5.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out5.csv", index=False, header=False)

############################################################## ROBIN

mm6 = []
for ii in range(0, 240):
    for jj in range(0, 320):
        blk6 = robinResized[ii:ii+16,jj:jj+16]
        flt6 = blk6.flatten()
        mm6.append(flt6)

mm6 = np.vstack(mm6)

df6 = pd.DataFrame(mm6)
df6[len(flt6)] = 2   # label 2 for robin

df6.to_csv("C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out6.csv", index=False, header=False)


#Create feature space using 16x16 block function
cardinal16x16 = createBlockFeatureVectors(cardinalResized, 'cardinal', '1')
sparrow16x16 = createBlockFeatureVectors(sparrowResized, 'sparrow', '2')
robin16x16 = createBlockFeatureVectors(robinResized, 'robin', '3')

#Create feature space using sliding block function
cardinalSlide = createSlidingFeatureVectors(cardinalResized, 'cardinal', '4')
sparrowSlide = createSlidingFeatureVectors(sparrowResized, 'sparrow', '5')
robinSlide = createSlidingFeatureVectors(robinResized, 'robin', '6')

######################## STATISTICAL MEAURES AND GRAPHS (16x16 Block)

cardinalDataSet1 = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out1.csv')
sparrowDataSet1 = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out2.csv')
robinDataSet1 = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out3.csv')

print("Number of cardinal features (16x16 block): ", featureCount(cardinalDataSet1))
print("Number of cardinal observations (16x16 block): ", observationCount(cardinalDataSet1))
print("Number of sparrow features (16x16 Block): ", featureCount(sparrowDataSet1))
print("Number of sparrow observations (16x16 block): ", observationCount(sparrowDataSet1))
print("Number of robin features (16x16 Block): ", featureCount(robinDataSet1))
print("Number of robin observations (16x16 block): ", observationCount(robinDataSet1), "\n")

cardinalHighDimensional1 = dimensionality(cardinalDataSet1)
sparrowHighDimensional1 = dimensionality(sparrowDataSet1)
robinHighDimensional1 = dimensionality(robinDataSet1)

if (cardinalHighDimensional1):
    print("The cardinal image data set (16x16 block) is high dimensional.")
else:
    print("The cardinal image data set (16x16 block) is not high dimensional.")
if (sparrowHighDimensional1):
    print("The sparrow image data set (16x16 block) is high dimensional.")
else:
    print("The sparrow image data set (16x16 block) is not high dimensional.")
if (robinHighDimensional1):
    print("The robin image data set (16x16 block) is high dimensional.\n")
else:
    print("The robin image data set (16x16 block) is not high dimensional.\n")

# Calculate and display mean values of features for each bird
cardinalMean1 = featureMeanDataset(cardinalDataSet1)
sparrowMean1 = featureMeanDataset(sparrowDataSet1)
robinMean1 = featureMeanDataset(robinDataSet1)
plt.plot(cardinalMean1, label = 'Cardinal')
plt.plot(sparrowMean1, label = 'Sparrow')
plt.plot(robinMean1, label = 'Robin')
plt.xlabel("Feature Numbers")
plt.ylabel("Mean Values")
plt.legend()
plt.show()

# Calculate and display the variance of the features for each bird
cardinalVariance1 = featureVarianceDataset(cardinalDataSet1)
sparrowVariance1 = featureVarianceDataset(sparrowDataSet1)
robinVariance1 = featureVarianceDataset(robinDataSet1)
plt.plot(cardinalVariance1, label = 'Cardinal')
plt.plot(sparrowVariance1, label = 'Sparrow')
plt.plot(robinVariance1, label = 'Robin')
plt.xlabel('Feature Numbers')
plt.ylabel('Variance')
plt.legend()
plt.show()

# Calculate and display standard deviations of features for each bird
cardinalStd1 = featureStandDevDataset(cardinalDataSet1)
sparrowStd1 = featureStandDevDataset(sparrowDataSet1)
robinStd1 = featureStandDevDataset(robinDataSet1)
plt.plot(cardinalStd1, label = 'Cardinal')
plt.plot(sparrowStd1, label = 'Sparrow')
plt.plot(robinStd1, label = 'Robin')
plt.xlabel("Feature Numbers")
plt.ylabel("Standard Deviation Values")
plt.legend()
plt.show()

######################## STATISTICAL MEAURES AND GRAPHS (Sliding Block)

cardinalDataSet2 = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out4.csv')
sparrowDataSet2 = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out5.csv')
robinDataSet2 = openFile('C:/Users/rober/OneDrive/Desktop/Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/out6.csv')

print("Number of cardinal features (sliding block): ", featureCount(cardinalDataSet2))
print("Number of cardinal observations (sliding block): ", observationCount(cardinalDataSet2))
print("Number of sparrow features (sliding block): ", featureCount(sparrowDataSet2))
print("Number of sparrow observations (sliding block): ", observationCount(sparrowDataSet2))
print("Number of robin features (sliding block): ", featureCount(robinDataSet2))
print("Number of robin observations (sliding block): ", observationCount(robinDataSet2), "\n")

cardinalHighDimensional2 = dimensionality(cardinalDataSet2)
sparrowHighDimensional2 = dimensionality(sparrowDataSet2)
robinHighDimensional2 = dimensionality(robinDataSet2)

if (cardinalHighDimensional2):
    print("The cardinal image data set (sliding block) is high dimensional.")
else:
    print("The cardinal image data set (sliding block) is not high dimensional.")
if (sparrowHighDimensional2):
    print("The sparrow image data set (sliding block) is high dimensional.")
else:
    print("The sparrow image data set (sliding block) is not high dimensional.")
if (robinHighDimensional2):
    print("The robin image data set (sliding block) is high dimensional.\n")
else:
    print("The robin image data set (sliding block) is not high dimensional.\n")

# Calculate and display mean values of features for each bird
cardinalMean2 = featureMeanDataset(cardinalDataSet2)
sparrowMean2 = featureMeanDataset(sparrowDataSet2)
robinMean2 = featureMeanDataset(robinDataSet2)
plt.plot(cardinalMean2, label = 'Cardinal')
plt.plot(sparrowMean2, label = 'Sparrow')
plt.plot(robinMean2, label = 'Robin')
plt.xlabel("Feature Numbers")
plt.ylabel("Mean Values")
plt.legend()
plt.show()

# Calculate and display the variance of the features for each bird
cardinalVariance2 = featureVarianceDataset(cardinalDataSet2)
sparrowVariance2 = featureVarianceDataset(sparrowDataSet2)
robinVariance2 = featureVarianceDataset(robinDataSet2)
plt.plot(cardinalVariance2, label = 'Cardinal')
plt.plot(sparrowVariance2, label = 'Sparrow')
plt.plot(robinVariance2, label = 'Robin')
plt.xlabel('Feature Numbers')
plt.ylabel('Variance')
plt.legend()
plt.show()

# Calculate and display standard deviations of features for each bird
cardinalStd2 = featureStandDevDataset(cardinalDataSet2)
sparrowStd2 = featureStandDevDataset(sparrowDataSet2)
robinStd2 = featureStandDevDataset(robinDataSet2)
plt.plot(cardinalStd2, label = 'Cardinal')
plt.plot(sparrowStd2, label = 'Sparrow')
plt.plot(robinStd2, label = 'Robin')
plt.xlabel("Feature Numbers")
plt.ylabel("Standard Deviation Values")
plt.legend()
plt.show()

#Feature Space construction using the 16x16 block data
cardinalSparrowMerged = mergeTwoFeatureVectors16(cardinalDataSet1, sparrowDataSet1, '1')
cardinalRobinMerged = mergeTwoFeatureVectors16(cardinalDataSet1, robinDataSet1, '2')
sparrowRobinMerged = mergeTwoFeatureVectors16(sparrowDataSet1, robinDataSet1, '3')
cardinalSparrowRobinMerged = mergeThreeFeatureVectors16(cardinalDataSet1, sparrowDataSet1, robinDataSet1, '1')

#Select three random features
a = 7
b = 11
c = 112

plt.scatter(cardinalSparrowMerged[:,a], cardinalSparrowMerged[:,b], c=cardinalSparrowMerged[:, 256], s=1)
plt.xlabel('Feature %i' % a)
plt.ylabel('Feature %i' % b)
plt.show()

plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
fig1 = ax.scatter3D(cardinalSparrowRobinMerged[:, a], cardinalSparrowRobinMerged[:, b], cardinalSparrowRobinMerged[:, c], s=5, c=cardinalSparrowRobinMerged[:, 256])
ax.set_xlabel('Feature %i' % a)
ax.set_ylabel('Feature %i' % b)
ax.set_zlabel('Feature %i' % c)
plt.show()

#Feature space construction using the sliding block data
cardinalSparrowMerged2 = mergeTwoFeatureVectorsSliding(cardinalDataSet2, sparrowDataSet2, '4')
cardinalRobinMerged2 = mergeTwoFeatureVectorsSliding(cardinalDataSet2, robinDataSet2, '5')
sparrowRobinMerged2 = mergeTwoFeatureVectorsSliding(sparrowDataSet2, robinDataSet2, '6')
cardinalSparrowRobinMerged2 = mergeThreeFeatureVectorsSliding(cardinalDataSet2, sparrowDataSet2,robinDataSet2, '2')

plt.scatter(cardinalSparrowMerged2[:,a], cardinalSparrowMerged2[:,b], c=cardinalSparrowMerged2[:, 256], s=1)
plt.xlabel('Feature %i' % a)
plt.ylabel('Feature %i' % b)
plt.show()

plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
fig1 = ax.scatter3D(cardinalSparrowRobinMerged2[:, a], cardinalSparrowRobinMerged2[:, b], cardinalSparrowRobinMerged2[:, c], s=5, c=cardinalSparrowRobinMerged2[:, 256])
ax.set_xlabel('Feature %i' % a)
ax.set_ylabel('Feature %i' % b)
ax.set_zlabel('Feature %i' % c)
plt.show()
'''
# Read merged datasets
cardinalSparrowMerged = pd.read_csv("C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image01-1.csv", header=None)
cardinalRobinMerged = pd.read_csv("C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image01-2.csv", header=None)
sparrowRobinMerged = pd.read_csv("C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image01-3.csv", header=None)
cardinalSparrowRobinMerged = pd.read_csv("C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image012-1.csv", header=None)

# Call to function that splits datasets 80% train and 20% test
CS_X1_80, CS_Y1_80, CS_X1_20, CS_Y1_20 = split8020dataset(256, cardinalSparrowMerged, 'CS_X', 'CS_Y')
CR_X1_80, CR_Y1_80, CR_X1_20, CR_Y1_20 = split8020dataset(256, cardinalRobinMerged, 'CR_X', 'CR_Y')
SR_X1_80, SR_Y1_80, SR_X1_20, SR_Y1_20 = split8020dataset(256, sparrowRobinMerged, 'SR_X', 'SR_Y')
CSR_X1_80, CSR_Y1_80, CSR_X1_20, CSR_Y1_20 = split8020dataset(256, cardinalSparrowRobinMerged, 'CSR_X', 'CSR_Y')
'''
#Feature selections
a = 204
b = 121
c = 7
d = 26


# Display histogram of two cardinal/sparrow features within train and test categories
plt.hist([CS_X1_80[:, a], CS_X1_80[:, b]], label=['Feature #%i'  %a, 'Feature #%i' %b])
plt.title('Train - 80% of cardinal and sparrow merged dataset')
plt.legend()
plt.show()
plt.hist([CS_X1_20[:, c], CS_X1_20[:, d]], label=['Feature #%i'  %c, 'Feature #%i' %d])
plt.title('Test - 20% of cardinal and sparrow merged dataset')
plt.legend()
plt.show()

print("The mean for feature", a, "is:", np.mean(CS_X1_80[:, a]), "in the cardinal/sparrow train dataset")
print("The mean for feature", b, "is:", np.mean(CS_X1_80[:, b]), "in the cardinal/sparrow train dataset")
print("The mean for feature", c, "is:", np.mean(CS_X1_80[:, c]), "in the cardinal/sparrow test dataset")
print("The mean for feature", d, "is:", np.mean(CS_X1_80[:, d]), "in the cardinal/sparrow test dataset")
print()


# Display histogram of two cardinal/robin features within train and test categories
plt.hist([CR_X1_80[:, a], CR_X1_80[:, b]], label=['Feature #%i'  %a, 'Feature #%i' %b])
plt.title('Train - 80% of cardinal and robin merged dataset')
plt.legend()
plt.show()
plt.hist([CR_X1_20[:, c], CR_X1_20[:, d]], label=['Feature #%i'  %c, 'Feature #%i' %d])
plt.title('Test - 20% of cardinal and robin merged dataset')
plt.legend()
plt.show()

print("The mean for feature", a, "is:", np.mean(CR_X1_80[:, a]), "in the cardinal/robin train dataset")
print("The mean for feature", b, "is:", np.mean(CR_X1_80[:, b]), "in the cardinal/robin train dataset")
print("The mean for feature", c, "is:", np.mean(CR_X1_80[:, c]), "in the cardinal/robin test dataset")
print("The mean for feature", d, "is:", np.mean(CR_X1_80[:, d]), "in the cardinal/robin test dataset")
print()

# Display histogram of two sparrow/robin features within train and test categories
plt.hist([SR_X1_80[:, a], SR_X1_80[:, b]], label=['Feature #%i'  %a, 'Feature #%i' %b])
plt.title('Train - 80% of sparrow and robin merged dataset')
plt.legend()
plt.show()
plt.hist([SR_X1_20[:, c], SR_X1_20[:, d]], label=['Feature #%i'  %c, 'Feature #%i' %d])
plt.title('Test - 20% of sparrow and robin merged dataset)')
plt.legend()
plt.show()

print("The mean for feature", a, "is:", np.mean(SR_X1_80[:, a]), "in the sparrow/robin train dataset")
print("The mean for feature", b, "is:", np.mean(SR_X1_80[:, b]), "in the sparrow/robin train dataset")
print("The mean for feature", c, "is:", np.mean(SR_X1_20[:, c]), "in the sparrow/robin test dataset")
print("The mean for feature", d, "is:", np.mean(SR_X1_20[:, d]), "in the sparrow/robin test dataset")
print()

# Display histogram of two cardinal/sparrow/robin features within train and test categories
plt.hist([CSR_X1_80[:, a], CSR_X1_80[:, b]], label=['Feature #%i'  %a, 'Feature #%i' %b])
plt.title('Train - 80% of cardinal, sparrow, and robin merged dataset')
plt.legend()
plt.show()
plt.hist([CSR_X1_20[:, c], CSR_X1_20[:, d]], label=['Feature #%i'  %c, 'Feature #%i' %d])
plt.title('Test - 20% of cardinal, sparrow, and robin merged dataset')
plt.legend()
plt.show()

print("The mean for feature", a, "is:", np.mean(CSR_X1_80[:, a]), "in the cardinal/sparrow/robin train dataset")
print("The mean for feature", b, "is:", np.mean(CSR_X1_80[:, b]), "in the cardinal/sparrow/robin train dataset")
print("The mean for feature", c, "is:", np.mean(CSR_X1_20[:, c]), "in the cardinal/sparrow/robin test dataset")
print("The mean for feature", d, "is:", np.mean(CSR_X1_20[:, d]), "in the cardinal/sparrow/robin test dataset")
print()

# Display scatter plots of two cardinal/sparrow features within train and test categories
plt.scatter(CS_X1_80[:, a], CS_X1_80[:, b], s=2, c=CS_Y1_80)
plt.xlabel('Feature %i' % a)
plt.ylabel('Feature %i' % b)
plt.show()
plt.scatter(CS_X1_20[:, c], CS_X1_20[:, d], s=2, c=CS_Y1_20)
plt.xlabel('Feature %i' % c)
plt.ylabel('Feature %i' % d)
plt.show()

# Display scatter plots of two cardinal/robin features within train and test categories
plt.scatter(CR_X1_80[:, a], CR_X1_80[:, b], s=2, c=CR_Y1_80)
plt.xlabel('Feature %i' % a)
plt.ylabel('Feature %i' % b)
plt.show()
plt.scatter(CR_X1_20[:, c], CR_X1_20[:, d], s=2, c=CR_Y1_20)
plt.xlabel('Feature %i' % c)
plt.ylabel('Feature %i' % d)
plt.show()

# Display scatter plots of two sparrow/robin features within train and test categories
plt.scatter(SR_X1_80[:, a], SR_X1_80[:, b], s=2, c=SR_Y1_80)
plt.xlabel('Feature %i' % a)
plt.ylabel('Feature %i' % b)
plt.show()
plt.scatter(SR_X1_20[:, c], SR_X1_20[:, d], s=2, c=SR_Y1_20)
plt.xlabel('Feature %i' % c)
plt.ylabel('Feature %i' % d)
plt.show()

# Display scatter plots of two sparrow/robin features within train and test categories
plt.scatter(CSR_X1_80[:, a], CSR_X1_80[:, b], s=2, c=CSR_Y1_80)
plt.xlabel('Feature %i' % a)
plt.ylabel('Feature %i' % b)
plt.show()
plt.scatter(CSR_X1_20[:, c], CSR_X1_20[:, d], s=2, c=CSR_Y1_20)
plt.xlabel('Feature %i' % c)
plt.ylabel('Feature %i' % d)
plt.show()
'''
######################################### CARDINAL AND SPARROW LASSO REGRESSION
# Training with 80% data.
CS_X1_train = CS_X1_80
CS_Y1_train = CS_Y1_80

lr1 = Lasso(alpha=1.0)
lr1.fit(CS_X1_train, CS_Y1_train)

# Testing with 20% data.
CS_X1_test = CS_X1_20
CS_y_test = CS_Y1_20

CS_yhat_test = lr1.predict(CS_X1_test)
CS_yhat_test = np.where(CS_yhat_test > 0.5, 1, 0)

#Save the actual and predicted values.
saveActualAndPredictedTest(CS_Y1_20, CS_yhat_test, 'CS_lasso')

# Confusion matrix analytics
CC1_test = confusion_matrix(CS_y_test, CS_yhat_test)
saveConfusionMatrix(CC1_test, 'CS_lasso')
ConfusionMatrixDisplay(CC1_test).plot()

TN = CC1_test[1,1]
FP = CC1_test[1,0]
FN = CC1_test[0,1]
TP = CC1_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("Lasso regression model with cardinal and sparrow merged dataset:")
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(CS_y_test, CS_yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(CS_y_test, CS_yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CS_y_test, CS_yhat_test))
print()

# Classification report from sklearn library
#print(metrics.classification_report(CS_y_test, CS_yhat_test))

########################################### CARDINAL AND ROBIN LASSO REGRESSION
#Training with 80% data.
CR_X1_train = CR_X1_80
CR_Y1_train = CR_Y1_80

lr2 = Lasso(alpha=1.0)
lr2.fit(CR_X1_train, CR_Y1_train)

#Testing with 20% data.
CR_X1_test = CR_X1_20
CR_y_test = CR_Y1_20

CR_yhat_test = lr2.predict(CR_X1_test)
CR_yhat_test = np.where(CR_yhat_test > 1, 2, 0)

#Save the actual and predicted values.
saveActualAndPredictedTest(CR_Y1_20, CR_yhat_test, 'CR_lasso')

# Confusion matrix analytics
CC2_test = confusion_matrix(CR_y_test, CR_yhat_test)
saveConfusionMatrix(CC2_test, 'CR_lasso')
ConfusionMatrixDisplay(CC2_test).plot()

TN = CC2_test[1,1]
FP = CC2_test[1,0]
FN = CC2_test[0,1]
TP = CC2_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("Lasso regression model with cardinal and robin merged dataset:")
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(CR_y_test, CR_yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(CR_y_test, CR_yhat_test, pos_label=2))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CR_y_test, CR_yhat_test, pos_label=2))
print()

# Classification report from sklearn library
#print(metrics.classification_report(CR_y_test, CR_yhat_test))

############################################ SPARROW AND ROBIN LASSO REGRESSION
#Training with 80% data.
SR_X1_train = SR_X1_80
SR_Y1_train = SR_Y1_80

lr3 = Lasso(alpha=1.0)
lr3.fit(SR_X1_train, SR_Y1_train)

#Testing with 20% data.
SR_X1_test = SR_X1_20
SR_y_test = SR_Y1_20

SR_yhat_test = lr3.predict(SR_X1_test)
SR_yhat_test = np.where(SR_yhat_test > 1.5, 2, 1)

#Save the actual and predicted values.
saveActualAndPredictedTest(SR_Y1_20, SR_yhat_test, 'SR_lasso')

CC3_test = confusion_matrix(SR_y_test, SR_yhat_test)
saveConfusionMatrix(CC3_test, 'SR_lasso')
ConfusionMatrixDisplay(CC3_test).plot()

TN = CC3_test[1,1]
FP = CC3_test[1,0]
FN = CC3_test[0,1]
TP = CC3_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("Lasso regression model with sparrow and robin merged dataset:")
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(SR_y_test, SR_yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(SR_y_test, SR_yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(SR_y_test, SR_yhat_test))
print()

# Classification report from sklearn library
#print(metrics.classification_report(SR_y_test, SR_yhat_test))

################################# CARDINAL, SPARROW, AND ROBIN LASSO REGRESSION
CSR_X1_train = CSR_X1_80
CSR_Y1_train = CSR_Y1_80

lr4 = Lasso(alpha=1.0)
lr4.fit(CSR_X1_train, CSR_Y1_train)

CSR_X1_test = CSR_X1_20
CSR_y_test = CSR_Y1_20

CSR_yhat_test = lr4.predict(CSR_X1_test)
CSR_yhat_test = np.where((CSR_yhat_test > 0.5) & (CSR_yhat_test < 1.5), 1, (np.where(CSR_yhat_test > 1.5, 2, 0)))

#Save the actual and predicted values.
saveActualAndPredictedTest(CSR_Y1_20, CSR_yhat_test, 'CSR_lasso')

CC4_test = confusion_matrix(CSR_y_test, CSR_yhat_test)
saveConfusionMatrix(CC4_test, 'CSR_lasso')
ConfusionMatrixDisplay(CC4_test).plot()

#Classifier 0
TN = CC4_test[1,1] + CC4_test[1,2] + CC4_test[2,1] + CC4_test[2,2]
FP = CC4_test[1,0] + CC4_test[2,0]
FN = CC4_test[0,1] + CC4_test[0,2]
TP = CC4_test[0,0]
#Classifier 1
# TN = CC4_test[0,0] + CC4_test[0,2] + CC4_test[2,0] + CC4_test[2,2]
# FP = CC4_test[0,1] + CC4_test[2,1]
# FN = CC4_test[1,0] + CC4_test[1,2]
# TP = CC4_test[1,1]
#Classifier 2
# TN = CC4_test[0,0] + CC4_test[0,1] + CC4_test[1,0] + CC4_test[1,1]
# FP = CC4_test[0,2] + CC4_test[1,2]
# FN = CC4_test[2,0] + CC4_test[2,1]
# TP = CC4_test[2,2]
#Classifiers combined
# TN = CC4_test[1,1] + CC4_test[1,2] + CC4_test[2,1] + CC4_test[2,2] + CC4_test[0,0] + CC4_test[0,2] + CC4_test[2,0] + CC4_test[2,2] +CC4_test[0,0] + CC4_test[0,1] + CC4_test[1,0] + CC4_test[1,1]
# FP = CC4_test[1,0] + CC4_test[2,0] + CC4_test[0,1] + CC4_test[2,1] +  CC4_test[0,2] + CC4_test[1,2]
# FN = CC4_test[0,1] + CC4_test[0,2] + CC4_test[1,0] + CC4_test[1,2] + CC4_test[2,0] + CC4_test[2,1]
# TP = CC4_test[0,0] + CC4_test[1,1] + CC4_test[2,2]

FPFN = FP+FN
TPTN = TP+TN

print("Lasso regression model with cardinal, sparrow, and robin merged dataset:")
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:", metrics.accuracy_score(CSR_y_test, CSR_yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(CSR_y_test, CSR_yhat_test, average='macro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CSR_y_test, CSR_yhat_test, average='macro'))
print()

# Classification report from sklearn library
#print(metrics.classification_report(CSR_y_test, CSR_yhat_test))

############################################ CARDINAL AND SPARROW RANDOM FOREST

rf1 = RandomForestClassifier(n_estimators=250)
rf1.fit(CS_X1_train, CS_Y1_train)

CS_RF_yhat_test = rf1.predict(CS_X1_test)
CS_RF_yhat_test = np.where(CS_RF_yhat_test > 0.5, 1, 0)

# Call to function that randomizes the parameters for the RandomForestClassifier
#CS_RF_yhat_test = randomForestBestParameters(CS_X1_train, CS_Y1_train, CS_X1_test)

#Save the actual and predicted values.
saveActualAndPredictedTest(CS_Y1_20, CS_RF_yhat_test, 'CS_rand_forest')

CC5_test = confusion_matrix(CS_y_test, CS_RF_yhat_test)
saveConfusionMatrix(CC5_test, 'CS_rand_forest')
ConfusionMatrixDisplay(CC5_test).plot()

TN = CC5_test[1,1]
FP = CC5_test[1,0]
FN = CC5_test[0,1]
TP = CC5_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("Random forest model with cardinal and sparrow merged dataset:")
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:", metrics.accuracy_score(CS_y_test, CS_RF_yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(CS_y_test, CS_RF_yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CS_y_test, CS_RF_yhat_test))
print()

# Classification report from sklearn library
#print(metrics.classification_report(CS_y_test, CS_RF_yhat_test))

############################################## CARDINAL AND ROBIN RANDOM FOREST

rf2 = RandomForestClassifier(n_estimators=250)
rf2.fit(CR_X1_train, CR_Y1_train)

CR_RF_yhat_test = rf2.predict(CR_X1_test)
#CR_yhat_test = np.where(CR_yhat_test > 1, 2, 0)

# Call to function that randomizes the parameters for the RandomForestClassifier
#CR_RF_yhat_test = randomForestBestParameters(CR_X1_train, CR_Y1_train, CR_X1_test)

#Save the actual and predicted values.
saveActualAndPredictedTest(CR_Y1_20, CR_RF_yhat_test, 'CR_rand_forest')

CC6_test = confusion_matrix(CR_y_test, CR_RF_yhat_test)
saveConfusionMatrix(CC6_test, 'CR_rand_forest')
ConfusionMatrixDisplay(CC6_test).plot()

TN = CC6_test[1,1]
FP = CC6_test[1,0]
FN = CC6_test[0,1]
TP = CC6_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("Random forest model with cardinal and robin merged dataset:")
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:", metrics.accuracy_score(CR_y_test, CR_RF_yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(CR_y_test, CR_RF_yhat_test, pos_label=2))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CR_y_test, CR_RF_yhat_test, pos_label=2))
print()

# Classification report from sklearn library
#print(metrics.classification_report(CR_y_test, CR_RF_yhat_test))

############################################### SPARROW AND ROBIN RANDOM FOREST

rf3 = RandomForestClassifier(n_estimators=250)
rf3.fit(SR_X1_train, SR_Y1_train)

SR_RF_yhat_test = rf3.predict(SR_X1_test)
#SR_RF_yhat_test = np.where(SR_RF_yhat_test > 1.5, 2, 1)

# Call to function that randomizes the parameters for the RandomForestClassifier
#SR_RF_yhat_test = randomForestBestParameters(SR_X1_train, SR_Y1_train, SR_X1_test)

#Save the actual and predicted values.
saveActualAndPredictedTest(SR_Y1_20, SR_RF_yhat_test, 'SR_rand_forest')

CC7_test = confusion_matrix(SR_y_test, SR_RF_yhat_test)
saveConfusionMatrix(CC7_test, 'SR_rand_forest')
ConfusionMatrixDisplay(CC7_test).plot()

TN = CC7_test[1,1]
FP = CC7_test[1,0]
FN = CC7_test[0,1]
TP = CC7_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("Random forest model with sparrow and robin merged dataset:")
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:", metrics.accuracy_score(SR_y_test, SR_RF_yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(SR_y_test, SR_RF_yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(SR_y_test, SR_RF_yhat_test))
print()

# Classification report from sklearn library
#print(metrics.classification_report(SR_y_test, SR_RF_yhat_test))

#################################### CARDINAL, SPARROW, AND ROBIN RANDOM FOREST

rf4 = RandomForestClassifier(n_estimators=250)
rf4.fit(CSR_X1_train, CSR_Y1_train)

CSR_RF_yhat_test = rf4.predict(CSR_X1_test)
#CSR_RF_yhat_test = np.where((CSR_RF_yhat_test > 0.5) & (CSR_RF_yhat_test < 1.5), 1, (np.where(CSR_RF_yhat_test > 1.5, 2, 0)))

# Call to function that randomizes the parameters for the RandomForestClassifier
#CSR_RF_yhat_test = randomForestBestParameters(CSR_X1_train, CSR_Y1_train, SCR_X1_test)

#Save the actual and predicted values.
saveActualAndPredictedTest(CSR_Y1_20, CSR_RF_yhat_test, 'CSR_rand_forest')

CC8_test = confusion_matrix(CSR_y_test, CSR_RF_yhat_test)
saveConfusionMatrix(CC7_test, 'CSR_rand_forest')
ConfusionMatrixDisplay(CC8_test).plot()

#Classifier 0
TN = CC8_test[1,1] + CC8_test[1,2] + CC8_test[2,1] + CC8_test[2,2]
FP = CC8_test[1,0] + CC8_test[2,0]
FN = CC8_test[0,1] + CC8_test[0,2]
TP = CC8_test[0,0]
#Classifier 1
# TN = CC8_test[0,0] + CC8_test[0,2] + CC8_test[2,0] + CC8_test[2,2]
# FP = CC8_test[0,1] + CC8_test[2,1]
# FN = CC8_test[1,0] + CC8_test[1,2]
# TP = CC8_test[1,1]
#Classifier 2
# TN = CC8_test[0,0] + CC8_test[0,1] + CC8_test[1,0] + CC8_test[1,1]
# FP = CC8_test[0,2] + CC8_test[1,2]
# FN = CC8_test[2,0] + CC8_test[2,1]
# TP = CC8_test[2,2]
#Classifiers combined
# TN = CC8_test[1,1] + CC8_test[1,2] + CC8_test[2,1] + CC8_test[2,2] + CC8_test[0,0] + CC8_test[0,2] + CC8_test[2,0] + CC8_test[2,2] +CC8_test[0,0] + CC8_test[0,1] + CC8_test[1,0] + CC8_test[1,1]
# FP = CC8_test[1,0] + CC8_test[2,0] + CC8_test[0,1] + CC8_test[2,1] + CC8_test[0,2] + CC8_test[1,2]
# FN = CC8_test[0,1] + CC8_test[0,2] + CC8_test[1,0] + CC8_test[1,2] + CC8_test[2,0] + CC8_test[2,1]
# TP = CC8_test[0,0] + CC8_test[1,1] + CC8_test[2,2]

FPFN = FP+FN
TPTN = TP+TN

print("Random forest model with cardinal, sparrow, and robin merged dataset:")
Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:", metrics.accuracy_score(CSR_y_test, CSR_RF_yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(CSR_y_test, CSR_RF_yhat_test, average='macro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CSR_y_test, CSR_RF_yhat_test, average='macro'))
print()

# Classification report from sklearn library
#print(metrics.classification_report(CSR_y_test, CSR_RF_yhat_test))

################################################## PRINCIPAL COMPONENT ANALYSIS

# Open the saved merged feature space files.
cardinalSparrowMerged = pd.read_csv("C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image01-1.csv", header=None)
cardinalRobinMerged = pd.read_csv("C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image01-2.csv", header=None)
sparrowRobinMerged = pd.read_csv("C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image01-3.csv", header=None)
cardinalSparrowRobinMerged = pd.read_csv("C:/Users/rober/OneDrive/Desktop\Spring Semester 2024/CSC 410 Big Data and Machine Learning/Assignment 3/Data/image012-1.csv", header=None)

# Standardize the feature spaces.
cardinalSparrowMergedStd = standardize(256, cardinalSparrowMerged, '1')
cardinalRobinMergedStd = standardize(256, cardinalRobinMerged, '2')
sparrowRobinMergedStd = standardize(256, sparrowRobinMerged, '3')
cardinalSparrowRobinMergedStd = standardize(256, cardinalSparrowRobinMerged, '4')

# Call to function that splits datasets 80% train and 20% test.
CS_X1_80_std, CS_Y1_80_std, CS_X1_20_std, CS_Y1_20_std = split8020dataset(256, cardinalSparrowMergedStd, 'CS_X_std', 'CS_Y_std')
CR_X1_80_std, CR_Y1_80_std, CR_X1_20_std, CR_Y1_20_std = split8020dataset(256, cardinalRobinMergedStd, 'CR_X_std', 'CR_Y_std')
SR_X1_80_std, SR_Y1_80_std, SR_X1_20_std, SR_Y1_20_std = split8020dataset(256, sparrowRobinMergedStd, 'SR_X_std', 'SR_Y_std')
CSR_X1_80_std, CSR_Y1_80_std, CSR_X1_20_std, CSR_Y1_20_std = split8020dataset(256, cardinalSparrowRobinMergedStd, 'CSR_X_std', 'CSR_Y_std')

#Call to function that splits datasets 80% train and 20% test for testing un-standardized dataset for testing 
# CS_X1_80_std, CS_Y1_80_std, CS_X1_20_std, CS_Y1_20_std = split8020dataset(256, cardinalSparrowMerged, 'CS_X', 'CS_Y')
# CR_X1_80_std, CR_Y1_80_std, CR_X1_20_std, CR_Y1_20_std = split8020dataset(256, cardinalRobinMerged, 'CR_X', 'CR_Y')
# SR_X1_80_std, SR_Y1_80_std, SR_X1_20_std, SR_Y1_20_std = split8020dataset(256, sparrowRobinMerged, 'SR_X', 'SR_Y')
# CSR_X1_80_std, CSR_Y1_80_std, CSR_X1_20_std, CSR_Y1_20_std = split8020dataset(256, cardinalSparrowRobinMerged, 'CSR_X', 'CSR_Y')


seed = 1
np.random.seed(seed)

cmp = 25

###################################################### CARDINAL AND SPARROW PCA
# Training with 80% data.
pca1 = PCA(n_components=cmp)
CS_X1_80_std = pca1.fit_transform(CS_X1_80_std)

# Using previous models for comparsion
lr_model1 = lr1.fit(CS_X1_80_std, CS_Y1_80_std)
rf_model1 = rf1.fit(CS_X1_80_std, CS_Y1_80_std)

# Testing with 20% data.
pca1 = PCA(n_components=cmp)
CS_X1_20_std = pca1.fit_transform(CS_X1_20_std)


# Using PCA to test lasso regression and random forest models on same dataset.
CS_LR_yhat_test_with_PCA = lr1.predict(CS_X1_20_std)
CS_LR_yhat_test_with_PCA = np.where(CS_LR_yhat_test_with_PCA > 0.5, 1, 0)
CS_RF_yhat_test_with_PCA = rf1.predict(CS_X1_20_std)

#Save the actual and predicted values.
saveActualAndPredictedTest(CS_Y1_20_std, CS_LR_yhat_test_with_PCA, 'CS_lasso_PCA')
saveActualAndPredictedTest(CS_Y1_20_std, CS_RF_yhat_test_with_PCA, 'CS_rand_forest_PCA')

# Confusion matrix analytics
CC9_test = confusion_matrix(CS_Y1_20_std, CS_LR_yhat_test_with_PCA)
saveConfusionMatrix(CC9_test, 'CS_lasso_PCA')
ConfusionMatrixDisplay(CC9_test).plot()
CC10_test = confusion_matrix(CS_Y1_20_std, CS_RF_yhat_test_with_PCA)
saveConfusionMatrix(CC10_test, 'CS_rand_forest_PCA')
ConfusionMatrixDisplay(CC10_test).plot()

#TN = CC_test[0,0]
#FP = CC_test[0,1]
#FN = CC_test[1,0]
#TP = CC_test[1,1]

TN = CC9_test[1,1]
FP = CC9_test[1,0]
FN = CC9_test[0,1]
TP = CC9_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("PCA and lasso regression model using cardinal/sparrow dataset")

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(CS_Y1_20_std, CS_LR_yhat_test_with_PCA))
print("BuiltIn_Precision:",metrics.precision_score(CS_Y1_20_std, CS_LR_yhat_test_with_PCA))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CS_Y1_20_std, CS_LR_yhat_test_with_PCA))
print("\n")
print(classification_report(CS_Y1_20_std, CS_LR_yhat_test_with_PCA))

TN = CC10_test[1,1]
FP = CC10_test[1,0]
FN = CC10_test[0,1]
TP = CC10_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("PCA and random forest model using cardinal/sparrow dataset")

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(CS_Y1_20_std, CS_RF_yhat_test_with_PCA))
print("BuiltIn_Precision:",metrics.precision_score(CS_Y1_20_std, CS_RF_yhat_test_with_PCA))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CS_Y1_20_std, CS_RF_yhat_test_with_PCA))
print("\n")
print(classification_report(CS_Y1_20_std, CS_RF_yhat_test_with_PCA))

CS_RF_yhat_test_with_PCA = rf_model1.predict_proba(CS_X1_20_std)
skplt.metrics.plot_roc(CS_Y1_20_std, CS_RF_yhat_test_with_PCA)
plt.title("ROC Curves - RF Classifier")
plt.show()

# Scatterplot of first two principal components
plt.scatter(CS_X1_80_std[:,0], CS_X1_80_std[:,1], c=CS_Y1_80_std)
plt.title('Principal component analysis: cardinal and sparrow dataset')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.title('Cardinal/sparrow PCA Variance Ratio')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

######################################################## CARDINAL AND ROBIN PCA
# Training with 80% data.
pca2 = PCA(n_components=cmp)
CR_X1_80_std = pca2.fit_transform(CR_X1_80_std)

# Using previous models for comparsion
lr_model2 = lr2.fit(CR_X1_80_std, CR_Y1_80_std)
rf_model2 = rf2.fit(CR_X1_80_std, CR_Y1_80_std)

# Testing with 20% data.
pca2 = PCA(n_components=cmp)
CR_X1_20_std = pca2.fit_transform(CR_X1_20_std)

# Using PCA to test lasso regression and random forest models on same dataset.
CR_LR_yhat_test_with_PCA = lr2.predict(CR_X1_20_std)
CR_LR_yhat_test_with_PCA = np.where(CR_LR_yhat_test_with_PCA > 1, 2, 0)
CR_RF_yhat_test_with_PCA = rf2.predict(CR_X1_20_std)

#Save the actual and predicted values.
saveActualAndPredictedTest(CR_Y1_20_std, CR_LR_yhat_test_with_PCA, 'CR_lasso_PCA')
saveActualAndPredictedTest(CR_Y1_20_std, CR_RF_yhat_test_with_PCA, 'CR_rand_forest_PCA')

# Confusion matrix analytics
CC11_test = confusion_matrix(CR_Y1_20_std, CR_LR_yhat_test_with_PCA)
saveConfusionMatrix(CC11_test, 'CR_lasso_PCA')
ConfusionMatrixDisplay(CC11_test).plot()
CC12_test = confusion_matrix(CR_Y1_20_std, CR_RF_yhat_test_with_PCA)
saveConfusionMatrix(CC12_test, 'CR_rand_forest_PCA')
ConfusionMatrixDisplay(CC12_test).plot()

#TN = CC_test[0,0]
#FP = CC_test[0,1]
#FN = CC_test[1,0]
#TP = CC_test[1,1]

TN = CC11_test[1,1]
FP = CC11_test[1,0]
FN = CC11_test[0,1]
TP = CC11_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("PCA and lasso regression model using cardinal/robin dataset")

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(CR_Y1_20_std, CR_LR_yhat_test_with_PCA))
print("BuiltIn_Precision:",metrics.precision_score(CR_Y1_20_std, CR_LR_yhat_test_with_PCA, pos_label=2))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CR_Y1_20_std, CR_LR_yhat_test_with_PCA, pos_label=2))
print("\n")
print(classification_report(CR_Y1_20_std, CR_LR_yhat_test_with_PCA))

#TN = CC_test[0,0]
#FP = CC_test[0,1]
#FN = CC_test[1,0]
#TP = CC_test[1,1]

TN = CC12_test[1,1]
FP = CC12_test[1,0]
FN = CC12_test[0,1]
TP = CC12_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("PCA and random forest model using cardinal/robin dataset")

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(CR_Y1_20_std, CR_RF_yhat_test_with_PCA))
print("BuiltIn_Precision:",metrics.precision_score(CR_Y1_20_std, CR_RF_yhat_test_with_PCA, pos_label=2))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CR_Y1_20_std, CR_RF_yhat_test_with_PCA, pos_label=2))
print("\n")
print(classification_report(CR_Y1_20_std, CR_RF_yhat_test_with_PCA))

CR_RF_yhat_test_with_PCA = rf_model2.predict_proba(CR_X1_20_std)
plot = skplt.metrics.plot_roc(CR_Y1_20_std, CR_RF_yhat_test_with_PCA)
plt.title("ROC Curves - RF Classifier")
plt.show()

# Scatterplot of first two principal components
plt.scatter(CR_X1_80_std[:,0], CR_X1_80_std[:,1], c=CR_Y1_80_std)
plt.title('Principal component analysis: cardinal and robin dataset')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

plt.plot(np.cumsum(pca2.explained_variance_ratio_))
plt.title('Cardinal/robin PCA Variance Ratio')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

######################################################### SPARROW AND ROBIN PCA
# Training with 80% data.
pca3 = PCA(n_components=cmp)
SR_X1_80_std = pca3.fit_transform(SR_X1_80_std)

# Using previous models for comparsion
lr_model3 = lr3.fit(SR_X1_80_std, SR_Y1_80_std)
rf_model3 = rf3.fit(SR_X1_80_std, SR_Y1_80_std)

# Testing with 20% data.
pca3 = PCA(n_components=cmp)
SR_X1_20_std = pca3.fit_transform(SR_X1_20_std)

# Using PCA to test lasso regression and random forest models on same dataset.
SR_LR_yhat_test_with_PCA = lr3.predict(SR_X1_20_std)
SR_LR_yhat_test_with_PCA = np.where(SR_LR_yhat_test_with_PCA > 1.5, 2, 1)
SR_RF_yhat_test_with_PCA = rf3.predict(SR_X1_20_std)

#Save the actual and predicted values.
saveActualAndPredictedTest(SR_Y1_20_std, SR_LR_yhat_test_with_PCA, 'SR_lasso_PCA')
saveActualAndPredictedTest(SR_Y1_20_std, SR_RF_yhat_test_with_PCA, 'SR_rand_forest_PCA')

# Confusion matrix analytics
CC13_test = confusion_matrix(SR_Y1_20_std, SR_LR_yhat_test_with_PCA)
saveConfusionMatrix(CC13_test, 'SR_lasso_PCA')
ConfusionMatrixDisplay(CC13_test).plot()
CC14_test = confusion_matrix(SR_Y1_20_std, SR_RF_yhat_test_with_PCA)
saveConfusionMatrix(CC14_test, 'SR_rand_forest_PCA')
ConfusionMatrixDisplay(CC14_test).plot()

#TN = CC_test[0,0]
#FP = CC_test[0,1]
#FN = CC_test[1,0]
#TP = CC_test[1,1]

TN = CC13_test[1,1]
FP = CC13_test[1,0]
FN = CC13_test[0,1]
TP = CC13_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("PCA and lasso regression model using sparrow/robin dataset")

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(SR_Y1_20_std, SR_LR_yhat_test_with_PCA))
print("BuiltIn_Precision:",metrics.precision_score(SR_Y1_20_std, SR_LR_yhat_test_with_PCA))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(SR_Y1_20_std, SR_LR_yhat_test_with_PCA))
print("\n")
print(classification_report(SR_Y1_20_std, SR_LR_yhat_test_with_PCA))

TN = CC14_test[1,1]
FP = CC14_test[1,0]
FN = CC14_test[0,1]
TP = CC14_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("PCA and random forest model using sparrow/robin dataset")

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(SR_Y1_20_std, SR_RF_yhat_test_with_PCA))
print("BuiltIn_Precision:",metrics.precision_score(SR_Y1_20_std, SR_RF_yhat_test_with_PCA))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(SR_Y1_20_std, SR_RF_yhat_test_with_PCA))
print("\n")
print(classification_report(SR_Y1_20_std, SR_RF_yhat_test_with_PCA))

SR_RF_yhat_test_with_PCA = rf_model3.predict_proba(SR_X1_20_std)
plot = skplt.metrics.plot_roc(SR_Y1_20_std, SR_RF_yhat_test_with_PCA)
plt.title("ROC Curves - RF Classifier")
plt.show()

# Scatterplot of first two principal components
plt.scatter(SR_X1_80_std[:,0], SR_X1_80_std[:,1], c=SR_Y1_80_std)
plt.title('Principal component analysis: sparrow and robin dataset')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

plt.plot(np.cumsum(pca3.explained_variance_ratio_))
plt.title('Sparrow/Robin PCA Variance Ratio')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

############################################## CARDINAL, SPARROW, AND ROBIN PCA
# Training with 80% data.
pca4 = PCA(n_components=cmp)
CSR_X1_80_std = pca4.fit_transform(CSR_X1_80_std)

# Using previous models for comparsion
lr_model4 = lr4.fit(CSR_X1_80_std, CSR_Y1_80_std)
rf_model4 = rf4.fit(CSR_X1_80_std, CSR_Y1_80_std)

# Testing with 20% data.
pca4 = PCA(n_components=cmp)
CSR_X1_20_std = pca4.fit_transform(CSR_X1_20_std)

# Using PCA to test lasso regression and random forest models on same dataset.
CSR_LR_yhat_test_with_PCA = lr4.predict(CSR_X1_20_std)
CSR_LR_yhat_test_with_PCA = np.where((CSR_LR_yhat_test_with_PCA > 0.5) & (CSR_LR_yhat_test_with_PCA < 1.5), 1, (np.where(CSR_LR_yhat_test_with_PCA > 1.5, 2, 0)))
CSR_RF_yhat_test_with_PCA = rf4.predict(CSR_X1_20_std)

#Save the actual and predicted values.
saveActualAndPredictedTest(CSR_Y1_20_std, CSR_LR_yhat_test_with_PCA, 'CSR_lasso_PCA')
saveActualAndPredictedTest(CSR_Y1_20_std, CSR_RF_yhat_test_with_PCA, 'CSR_rand_forest_PCA')

# Confusion matrix analytics
CC15_test = confusion_matrix(CSR_Y1_20_std, CSR_LR_yhat_test_with_PCA)
saveConfusionMatrix(CC15_test, 'CSR_lasso_PCA')
ConfusionMatrixDisplay(CC15_test).plot()
CC16_test = confusion_matrix(CSR_Y1_20_std, CSR_RF_yhat_test_with_PCA)
saveConfusionMatrix(CC16_test, 'CSR_rand_forest_PCA')
ConfusionMatrixDisplay(CC16_test).plot()

#TN = CC_test[0,0]
#FP = CC_test[0,1]
#FN = CC_test[1,0]
#TP = CC_test[1,1]

TN = CC15_test[1,1] + CC15_test[1,2] + CC15_test[2,1] + CC15_test[2,2]
FP = CC15_test[1,0] + CC15_test[2,0]
FN = CC15_test[0,1] + CC15_test[0,2]
TP = CC15_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("PCA and lasso regression model using cardinal/sparrow/robin dataset")

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(CSR_Y1_20_std, CSR_LR_yhat_test_with_PCA))
print("BuiltIn_Precision:",metrics.precision_score(CSR_Y1_20_std, CSR_LR_yhat_test_with_PCA, average='macro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CSR_Y1_20_std, CSR_LR_yhat_test_with_PCA, average='macro'))
print("\n")
print(classification_report(CSR_Y1_20_std, CSR_LR_yhat_test_with_PCA))

TN = CC16_test[1,1] + CC16_test[1,2] + CC16_test[2,1] + CC16_test[2,2]
FP = CC16_test[1,0] + CC16_test[2,0]
FN = CC16_test[0,1] + CC16_test[0,2]
TP = CC16_test[0,0]

FPFN = FP+FN
TPTN = TP+TN

print("PCA and random forest model using cardinal/sparrow/robin dataset")

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)

print("BuiltIn_Accuracy:",metrics.accuracy_score(CSR_Y1_20_std, CSR_RF_yhat_test_with_PCA))
print("BuiltIn_Precision:",metrics.precision_score(CSR_Y1_20_std, CSR_RF_yhat_test_with_PCA, average='macro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(CSR_Y1_20_std, CSR_RF_yhat_test_with_PCA, average='macro'))
print("\n")
print(classification_report(CSR_Y1_20_std, CSR_RF_yhat_test_with_PCA))

CSR_RF_yhat_test_with_PCA = rf_model4.predict_proba(CSR_X1_20_std)
plot = skplt.metrics.plot_roc(CSR_Y1_20_std, CSR_RF_yhat_test_with_PCA)
plt.title("ROC Curves - RF Classifier")
plt.show()

# Scatterplot of first two principal components
plt.scatter(CSR_X1_80_std[:,0], CSR_X1_80_std[:,1], c=CSR_Y1_80_std)
plt.title('Principal component analysis: cardinal, sparrow, and robin dataset')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

plt.plot(np.cumsum(pca4.explained_variance_ratio_))
plt.title('Cardinal/sparrow/robin PCA Variance Ratio')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()




