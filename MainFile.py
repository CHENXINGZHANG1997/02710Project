'''
Created on Apr 12, 2013

@author: Bhushan Ramnani
'''
import numpy as np
import xlrd as xl
#import GERPKernel as ger
import PyML as ml
import DataImport as DI
import crossVal as cv



def generateKernelMatrix(DataSet):
    """Creates a kernel matrix/gram matrix from an input dataset, which is a list of examples"""
    
    n_samples = len(DataSet)
    kernelMatrix = np.empty([n_samples,n_samples], dtype = "string")
    PatternIds = np.empty([n_samples,1],dtype = "string")
    Labels = np.empty([n_samples,1],dtype = "string") 
    for i in xrange(n_samples):
        (label, tps, pID, A1) = DataSet[i]
        PatternIds[i,0] =  pID
        Labels[i,0] = label
    
    for i in xrange(n_samples):
        for j in xrange(n_samples):
            (label1, tps, pID, A1) = DataSet[i]
            (label2, tps, pID, A2) = DataSet[j]
            kernelMatrix[i,j] = str(ger.GERPKernel(A1,A2))
    
    kernelFileMatrix = np.concatenate(PatternIds, kernelMatrix)
    labelMatrix = np.concatenate(PatternIds, Labels)
    
    np.savetxt("labelText.txt", labelMatrix, delimiter = ',')
    np.savetxt("kernelText.txt", kernelFileMatrix,delimiter=',')
    
    labels = ml.Labels("labelText.txt")
    kdata = ml.kernelData("kernelText.txt")
    kdata.attachLabels(labels)
    
    return kdata





def main():
    wb = xl.open_workbook("Dataset_S1.xls")
    S = wb.sheet_by_index(0)
    allData = DI.get_patient_ts(S, 7, 6)
    #for (label, time, patient,ts) in allData:
    #    if(patient == str(1185163.0)):
    #        print ts 
    allData = DI.remove_missing_tp(allData, S, 7, 6)
    #for (label, time, patient,ts) in allData:
        #if(patient == str(1185163.0)):
            #print ts 
    #print allData[0]
    seg_data = cv.gen_time_split_data(allData, 4, 2, 5)
    for val in seg_data:
        (train, test) = val
        print ("This is new iteration")
        for(label, time, patient, ts) in train:
            print label, time, patient, len(ts)
        print("This is new dataset")
        for(label, time, patient, ts) in test:
            print label, time, patient, len(ts)
           #print "\n"
    #for(label, time, patient, ts) in test:
    #    print label, time, patient
    #    print "\n"
    #kdata = generateKernelMatrix(allData)
    




main()
#total = len(allData)
#n = int(0.7*total)
#TrainingData = allData[0:n]
#TestData = allData[n:total]
#s = ml.svm.SVM()

#(label, patientId, []list of time points, numpy matrix of time series)

    
    
        







        