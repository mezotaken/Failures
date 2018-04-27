import cv2
import csv
subdir = 'resized/'
with open('clear_full.csv', 'r') as fin:
        reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
        for i in range(0,5000):
            if(i%50 == 0):
                print(i/5000)
            path,gender = next(reader)
            img = cv2.imread(path)
            resized = cv2.resize(img,(512,512))
            #print(subdir+path)
            cv2.imwrite(subdir+path,resized)