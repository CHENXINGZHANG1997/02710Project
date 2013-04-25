'''
Created on Apr 13, 2013

@author: Bhushan Ramnani (modularized by Stuti Agrawal)
'''
import sys
import numpy as np
import xlrd as xl
import copy


#Time points are taken in four 3-month intervals and two 6-month intervals
#over two years.
def timePointToIndex(tp):
    if tp == 0: return 0
    elif tp == 3: return 1
    elif tp == 6: return 2
    elif tp == 9: return 3
    elif tp == 12: return 4
    elif tp == 18: return 5
    elif tp == 24: return 6
    else: 
        sys.exit("Invalid Time Point")


def get_patient_ts(S, time_points, ignore_cols): 
    """Takes an excel workbook sheet as input and segregates training examples as time series data, requires number of columns and number of time points"""
    
    TrainingExamples = []
    flag = True #Determines if the next patient is different or same.
    for i in xrange(1,S.nrows):#Ignore last row
        if flag:
            mat = np.zeros([time_points,S.ncols-ignore_cols], dtype = "float")
            label = str(S.cell(i,0).value)
            timePoints = []
            flag = False
        timePoint = str(S.cell(i,2).value)
        k = timePointToIndex(int(timePoint[1:]))
        timePoints.append(k)
    
        #Assign the expression values at each time point
        for j in xrange(S.ncols - ignore_cols):
            x = str(S.cell(i,j + ignore_cols).value)
            if x == "na": #If the expression value is unknown
                mat[k,j] = float(0)
            else:
                mat[k,j] = float(x)
        
        if i!=S.nrows-1:    
            PatientCode1 = str(S.cell(i,1).value)
            PatientCode2 = str(S.cell(i+1,1).value)
            if PatientCode1 != PatientCode2:
                flag = True             
                TrainingExamples.append((label, timePoints, PatientCode1,  mat))
        else:
            TrainingExamples.append((label, timePoints, PatientCode1,  mat))
            
    return TrainingExamples



def remove_missing_tp(TrainingExamples,S, time_points, ignore_cols):
    """Remove missing timepoints from Time series data. Any column with all zeros will be removed"""
    
    new_training_examples = list()
    #Delete the missing time points rows
    allzeros = np.zeros((S.ncols - ignore_cols,), dtype = 'float')
    for (label,tps, patient, A) in TrainingExamples:
        (row, col) = A.shape
        B = np.array([allzeros])
        for i in xrange(row):
            if np.any(A[i]):
                row = (np.array([A[i]]))
                if(not(np.any(B[0]))):
                    B[0] = row
                else:
                    B = np.concatenate((B,row), axis = 0)
        new_training_examples.append((label,tps,patient,B))
    return new_training_examples
