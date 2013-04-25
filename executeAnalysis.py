'''
Created on Apr 12, 2013

@author: Bhushan Ramnani
'''


import numpy as np
import xlrd as xl
import GERPKernel as ger
import PyML as ml
import DataImport as DI
import crossVal as cv
import TWED as twed
import copy
import math
import sys



def generateKernelMatrix(TestData, TrainingData, sigma = 10, lam = math.pow(10, -3), nu = 0.5, kernel = "twed"):
    """Creates a kernel matrix/gram matrix from an input dataset, which is a list of examples"""
    
    n_samplesTest = len(TestData)
    n_samplesTrain = len(TrainingData)
    kernelMatrix = np.empty([n_samplesTest,n_samplesTrain], dtype = "float")
    PatternIds = np.empty([n_samplesTest,1],dtype = object)
    Labels = np.empty([n_samplesTest,1],dtype = object) 
    for i in xrange(n_samplesTest):
        (label, tps, pID, A1) = TestData[i]
        PatternIds[i,0] =  pID
        Labels[i,0] = label
    
    for i in xrange(n_samplesTest):
        for j in xrange(n_samplesTrain):
            (label1, tps, pID, A1) = TestData[i]
            (label2, tps, pID, A2) = TrainingData[j]
            if(kernel == "gerp"):
                kernelMatrix[i,j] = str(ger.GERPKernel(A1,A2, sigma))
            elif(kernel == "twed"):
                kernelMatrix[i,j] = str(twed.TwedKernel(TestData[i], TrainingData[j], lam, nu, sigma)) 
    
    kernelFileMatrix = np.concatenate((PatternIds, kernelMatrix), axis=1)
    labelMatrix = np.concatenate((PatternIds, Labels), axis=1)
    
    np.savetxt("labelText.txt", labelMatrix, fmt = '%s', delimiter = ',')
    np.savetxt("kernelText.txt", kernelFileMatrix, fmt = "%s", delimiter=',')
    
    f1 = "labelText.txt"
    f2 = "kernelText.txt"
    
    labels = ml.Labels(f1)
    
    kdata = ml.KernelData(f2)
    kdata.attachLabels(labels)
    return kdata


def createVectorDataSet(A):
    """Converts a traing example list to a vector dataset"""
    
    labels= []
    patterns = []
    X = []
    for (label,tps,pID, B) in A:
        labels.append(label)
        patterns.append(pID)
        X.append(B)
    data = ml.VectorDataSet(X,L=labels, patternID=patterns)
    return data
        


def trainAndTest(TrainingData, TestData, C = 10, sigma =10, lam = math.pow(10, -3), nu = 0.5, kernel = "twed"):
    """Takes Training and test data. Returns the accuracy"""
    trdata = generateKernelMatrix(TrainingData, TrainingData, sigma, lam, nu, kernel)
    tedata = generateKernelMatrix(TestData, TrainingData, sigma, lam, nu, kernel)
    s = ml.svm.SVM(C= C)
    s.train(trdata)    
    r = s.test(tedata)
    print "Success Rate = ", r.getSuccessRate()
    return r.getSuccessRate()


def validationExperiment(allData, N = 5, F=4, c = 10, sigma = 10, lam = math.pow(10, -3), nu = 0.5, kernel = "twed"):
    """Takes dataset in out format and returns the average accuracy after N times F fold cross validation"""
    trdata = generateKernelMatrix(allData, allData, sigma, lam, nu, kernel)
    X = list()
    for i in xrange(N):
        s = ml.svm.SVM(C =c)
        r = s.cv(trdata, numFolds=F)
        X.append(r.getSuccessRate())        
    return np.mean(X)
        
        
def chooseOptimalModel(allData, kernel):
    """Takes the validation set and tests it against various values of C and sigma. Returns the most optimal parameters for which the average accuracy is maximum"""
    """Model Selection"""
    max = 0.0
    optC = 0.0
    optSigma = 0.0
    
    for i in xrange(4,-1,-1):
        for j in xrange(3,-4,-1):
            C = math.pow(10, i)
            sigma = math.pow(10, j)
            print "C = ", C
            print "sigma = ", sigma
            lam = math.pow(10, -1*i)
            nu = 0.5
            accuracy =  validationExperiment(allData,5,4, C, sigma, lam, nu, kernel)
            print accuracy
            if max<accuracy:
                max = accuracy
                optC = C
                optSigma = sigma
    
    
    #Only for twed kernel
    #Once you have the maximum accuracy for a particular value of C and sigma, we can look for the most optimal value of mu and lambda
    
    
    if kernel == "twed":
        lambdaValues = [0,0.25,0.5,0.75,1.0]
        optNu  = 0.0
        optLam = 0.0
        for i in xrange(-5,1):
            for l in lambdaValues:
                lam = l
                nu  = math.pow(10, i)
                accuracy =  validationExperiment(allData,5,4, optC, optSigma, lam, nu, kernel)
                if max<accuracy:
                    max = accuracy
                    optNu = nu
                    optLam = lam
                    optSigma = sigma
                
    if kernel=="twed":
        return optC, optSigma, max, optNu, optLam
    
    else:
        return optC, optSigma, max
        
    

def normalize(A):
    """Takes a time series data and normalizes it"""
    (x,y) = A.shape
    for i in xrange(x):
        mean = np.mean(A[i,:])
        stdDev = np.std(A[i,:])
        for j in xrange(y):
            A[i,j] = (A[i,j]-mean)/stdDev
             
def normalizeValues(allData):
    """Takes a list of time series examples and normalizes each examples"""
    for (label, tps, pID, A) in allData:
        normalize(A)
        

def main():
    """ Data import and analysis is performed using the two kernels. The analysis is stored in a text file."""
    
    
    wb = xl.open_workbook(sys.argv[1]) #retrice dataset from the excel file
    S = wb.sheet_by_index(0) #dataset sheet
    allData = DI.get_patient_ts(S, 7, 6) #Convert import data to our format data structure
    allData = DI.remove_missing_tp(allData, S, 7, 6) #remove missing time points from the time series data
    
       
    normalizeValues(allData) #normalize data
    
    kernel = sys.argv[2]
    #Analysis for TWED kernel
    if kernel == "twed":
        Analysis = {} #Pass the key in format "twedt1". Returns the most opimal parameters as a tuple of C,S,acc,Nu and Lambda
        f = open("analysisTWED.txt", "r+")
        for i in xrange(6):
            data = cv.gen_time_split_data(allData, i) #split the time series data
            key = kernel+"T"+str(i)
            Analysis[key] = chooseOptimalModel(data, kernel) #perform model selection 
            strToWrite = key+":   "+str(Analysis[key][0])+"   "+str(Analysis[key][1])+"   "+str(Analysis[key][2])+"   "+str(Analysis[key][3])+"   "+str(Analysis[key][4])+"\n"
            f.write(strToWrite)
        f.close()
    else:

    #Analysis for GERP kernel
        Analysis = {} #Pass the key in format "twedt1". Returns the most opimal parameters as a tuple of C,S,acc
        f = open("analysisGERP.txt", "r+")
        for i in xrange(6):
            data = cv.gen_time_split_data(allData, i)
            key = kernel+"T"+str(i)
            Analysis[key] = chooseOptimalModel(data, kernel) #perform model selection
            strToWrite = key+":   "+str(Analysis[key][0])+"   "+str(Analysis[key][1])+"   "+str(Analysis[key][2])+"\n"
            f.write(strToWrite)
        f.close()
    
    
    
    
    
main()


    
    
        







        