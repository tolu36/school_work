import numpy
import math
import random
#import matplotlib.pyplot
import sklearn.discriminant_analysis
from sklearn.decomposition import FastICA

# create data from a multivariate normal distribution
mean1 = [-3, 0, -1, -2, -4]
mean2 = [3, 1, 0, -2, 4]

cov = [[12,1,2,3,0],[1,13,0,0,0],[2,0,4,0,0],[3,0,0,5,0],[0,0,0,0,6]]
#guarantee that the matrix is symmtric and positive semidefinite
cov = 0.5*(cov + numpy.transpose(cov))
x1 = numpy.random.multivariate_normal(mean1, cov, 1000)
#matplotlib.pyplot.scatter(x1[:,0], x1[:,1], c = 'b', marker = '.')
x2 = numpy.random.multivariate_normal(mean2, cov, 1000)
#matplotlib.pyplot.scatter(x2[:,0], x2[:,1], c = 'r', marker = '.')
#matplotlib.pyplot.show()

X = numpy.concatenate((x1,x2))

#first half of the data from class 0, second half from class 1
Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))

#training set result without feature selection...
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X,Xc)
prediction = lda.predict(X)
error = sum(prediction != Xc)
print("total error with all features = ", error)

#use feature selection
#initialize the best features to a value that is invalid as a feature index
#this is a forward search

dimension = 2
#list of features selected by index
selected = []
#list of features remaining by index
remaining = [0,1,2,3,4]
#currently selected features from dataset
Xselection = numpy.empty((2000,0))

#iterate over all selected features
for iteration in range(dimension):
    
    #iterate over remaining features
    error = 10000*numpy.ones(5)
    for i in remaining:
            
        #now, add this to the previously selected features
        Xtest = Xselection
        Xtest = numpy.append(Xtest, X[:,i].reshape(2000,1), axis=1 )
       
        #classify the training data using currently selected features
        lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        lda.fit(Xtest,Xc)
        prediction = lda.predict(Xtest)
            
        error[i] = sum(prediction != Xc)

    #get the index of the best feature
    best = numpy.argmin(error)
    #update the selected feature list
    selected.append(best)
    #update the remaining feature list
    remaining.remove(best)
    #update the currently selected features from the database
    Xselection = numpy.append(Xselection, X[:,best].reshape(2000,1), axis=1)


#training set result with feature selection...
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(Xselection,Xc)
prediction = lda.predict(Xselection)
error = sum(prediction != Xc)
print("total feature selection error with two features = ", error)


# PCA
Xmc = X - numpy.mean(X)
D,E = numpy.linalg.eig(numpy.dot(Xmc.T,Xmc))

sortIndex = numpy.flip(numpy.argsort(D))

#ESorted = numpy.zeros((5,5))
ESorted = numpy.empty((5,0))
dimension = 5
for i in range(dimension):
    ESorted = numpy.append(ESorted, E[:,sortIndex[i]].reshape(5,1), axis=1)


meanSquareError = numpy.zeros(5,)
classificationError = numpy.zeros(5,)
ySorted = numpy.dot(X,ESorted)
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
for numDims in range(1,6): 
    
    # reconstruction
    yReduced = ySorted[:,0:numDims]
    EReduced = ESorted[:,0:numDims]
    XReconstructed = numpy.dot(yReduced, numpy.transpose(EReduced))
    meanSquareError[numDims-1] = sum(sum((XReconstructed - X)**2))/2000

    # classification
    #training
    lda.fit(yReduced,Xc)
    #testing
    prediction = lda.predict(yReduced)
    classificationError[numDims-1] = sum(prediction != Xc) # sum(prediction != Xc)

print("total pca error with two features = ", classificationError[1])


#ica
ica = FastICA(max_iter = 10000, tol = 0.001)
result = ica.fit_transform(X)
X = result


#use feature selection for ICA
#initialize the best features to a value that is invalid as a feature index
#this is a forward search

dimension = 2
#list of features selected by index
selected = []
#list of features remaining by index
remaining = [0,1,2,3,4]
#currently selected features from dataset
Xselection = numpy.empty((2000,0))

#iterate over all selected features
for iteration in range(dimension):
    
    #iterate over remaining features
    error = 10000*numpy.ones(5)
    for i in remaining:
            
        #now, add this to the previously selected features
        Xtest = Xselection
        Xtest = numpy.append(Xtest, X[:,i].reshape(2000,1), axis=1 )
       
        #classify the training data using currently selected features
        lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        lda.fit(Xtest,Xc)
        prediction = lda.predict(Xtest)
            
        error[i] = sum(prediction != Xc)

    #get the index of the best feature
    best = numpy.argmin(error)
    #update the selected feature list
    selected.append(best)
    #update the remaining feature list
    remaining.remove(best)
    #update the currently selected features from the database
    Xselection = numpy.append(Xselection, X[:,best].reshape(2000,1), axis=1)


#training set result for ica...
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(Xselection,Xc)
prediction = lda.predict(Xselection)
error = sum(prediction != Xc)
print("total ica error with two features = ", error)




print("done")





