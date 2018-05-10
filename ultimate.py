import numpy as np
import cv2
import csv
dictsize = 25
imageset = 1569
skip = 200 

per5 = int(imageset/10)

BOW = cv2.BOWKMeansTrainer(dictsize)
siftdet = cv2.xfeatures2d.SIFT_create()

print('Keypoints detecting, computing and preparing BOW')
with open('result.csv', 'r') as fin:
    reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
    for i in range(0,imageset):
        if(i%per5 == 0):
            print(i/imageset)
        path,gender = next(reader)
        img = cv2.imread(path)
        _,dsc = siftdet.compute(img,siftdet.detect(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),None))
        BOW.add(dsc)
print("Ready")

#Кластеризация, запись словаря
print('Dictionary clusterisation and saving')
dictionary = BOW.cluster()
print("Ready")

#Извлечение и запись признаков
siftext = cv2.xfeatures2d.SIFT_create()
bow_extract = cv2.BOWImgDescriptorExtractor(siftext,cv2.BFMatcher(cv2.NORM_L2))
bow_extract.setVocabulary(dictionary)

def feature_extract(pth):
    img = cv2.imread(pth)
    return  bow_extract.compute(img,siftdet.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

print("Feature extraction")
with open('result.csv', 'r') as fin:
    reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
    train_desc = []
    responses = []
    testresp = []
    testdesc = []
    for i in range(0,skip):
        path,gender = next(reader)
        testresp.append(gender)
        testdesc.extend(feature_extract(path))
        
    for i in range(skip,imageset):
        path,gender = next(reader)
        train_desc.extend(feature_extract(path))
        responses.append(gender)
print("Ready")  

train_desc=np.array(train_desc)
responses = np.int32(responses)
testresp = np.int32(testresp)
testdesc = np.float32(testdesc)
#print(responses)
predictor = cv2.ml.RTrees_create()
predictor.train(train_desc, cv2.ml.ROW_SAMPLE,responses )


count = 0
_,r = predictor.predict(testdesc)
for i in range(0,skip):
    if(testresp[i] == int(r[i][0])):
        count = count +1
 
print(count/skip)