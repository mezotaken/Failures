import numpy as np
import cv2
import csv
skip = 100
filename = 'ds25_is500.csv'
with open(filename, 'r') as fin:
    reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
    dictsize,imageset = next(reader)
    dictsize = int(dictsize)
    imageset = int(imageset)
    testresp = []
    testdesc = []
    for i in range(0,skip):
        testresp.append(next(reader)[0])
        testdesc.append(next(reader))
    testresp = np.int32(testresp)
    testdesc = np.float32(testdesc)
   
svm = cv2.ml.RTrees_load('test 100 of 500.xml')
res = 0
print(testresp)
for i in range(0,skip):
    if(testresp[i] == svm.predict(np.float32([testdesc[i]]))[0]):
        res = res+1
print(res)