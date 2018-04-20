import cv2
import csv
dictsize = 25
imageset = 500

per5 = int(imageset/20)
filename = 'ds' + str(dictsize) + '_is' + str(imageset) + '.csv'
with open(filename,'w',newline="") as fout:
    
    #Запись размера словаря, количества изображений
    writer = csv.writer(fout,quoting=csv.QUOTE_NONNUMERIC)
    row = (dictsize,imageset)
    writer.writerow(row)
    
    #Создание BOW для словаря
    BOW = cv2.BOWKMeansTrainer(dictsize)
    siftdet = cv2.xfeatures2d.SIFT_create()
    
    #Чтение из файлов, нахождение ключевых точек, заполнение BOW
    print('Keypoints detecting, computing and preparing BOW')
    with open('clear_full.csv', 'r') as fin:
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
    with open('clear_full.csv', 'r') as fin:
        reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
        for i in range(0,imageset):
            if(i%per5 == 0):
                print(i/imageset)
            path,gender = next(reader)
            writer.writerow([gender])
            writer.writerow(feature_extract(path)[0])
    print("Ready")
