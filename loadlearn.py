import numpy as np
import cv2
import csv
filename = 'ds100_is500.csv'
skip = 50
with open(filename, 'r') as fin:
    reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
    dictsize,imageset = next(reader)
    dictsize = int(dictsize)
    imageset = int(imageset)
    trainresp = []
    traindesc = []
    
    testresp = []
    testdesc = []
    for i in range(0,skip):
        testresp.append(next(reader)[0])
        testdesc.append(np.float32(next(reader)))
    
    for i in range(skip,imageset):
        trainresp.append(next(reader)[0])
        traindesc.append(np.float32(next(reader)))
    #print(trainresp)
    trainresp = np.int32(trainresp)
    traindesc = np.float32(traindesc)
    testresp = np.int32(testresp)
    testdesc = np.float32(testdesc)
    
    predictor = cv2.ml.RTrees_create()
    #predictor.setKernel(cv2.ml.SVM_SIGMOID)
    predictor.setMaxDepth(100)
    predictor.setMinSampleCount(100)
    #help(predictor)
    #print(predictor.getDegree())
    
    predictor.train(traindesc,cv2.ml.ROW_SAMPLE,trainresp)
    res = 0

    _,r = predictor.predict(testdesc)
    #print(r)
    matr = np.zeros((2,2))
    for i in range(0,skip):
        if(r[i] == 1):
            if(r[i] == testresp[i]):
                matr[0,0] = matr[0,0] + 1
            else:
                matr[1,0] = matr[1,0] + 1
        else:
            if(r[i] == testresp[i]):
                matr[1,1] = matr[1,1] + 1
            else:
                matr[0,1] = matr[0,1] + 1
    matr = np.divide(matr,skip)
 
    print(matr)
    
 