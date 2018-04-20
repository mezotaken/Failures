import numpy as np
import cv2
import csv
filename = 'ds25_is500.csv'
skip = 100
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
        next(reader)
        next(reader)
    
    for i in range(skip,imageset):
        trainresp.append(next(reader)[0])
        traindesc.append(next(reader))
    trainresp = np.int32(trainresp)
    print(trainresp)
    traindesc = np.float32(traindesc)
    
    predictor = cv2.ml.RTrees_create()
    predictor.train(traindesc,cv2.ml.ROW_SAMPLE,trainresp)
    name = 'test '+str(skip)+' of '+str(imageset)+'.xml'
    predictor.save(name);
    
    
 