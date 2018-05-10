import cv2
import csv
with open('clear_full.csv', 'r') as fin:
        reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
        filename = 'filtered.csv'
        with open(filename,'w',newline="") as fout:
            writer = csv.writer(fout,quoting=csv.QUOTE_NONNUMERIC)
            count = 0
            for i in range(0,2133): 
                next(reader)
            for i in range(2133,43239): 
                path,gender = next(reader)
                img = cv2.imread(path)
                if(gender == 1):
                    cv2.putText(img,'M',(10,50),  cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
                else:
                    cv2.putText(img,'F',(10,50),  cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow('image',img)
                cv2.moveWindow('image',0,0)
                retkey = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if(retkey == 32):
                    row = (path,gender)
                    writer.writerow(row)
                    count = count + 1
                if(retkey == 114):
                    gender = 1.0 - gender
                    row = (path,gender)
                    writer.writerow(row)
                    count = count + 1
                if(retkey == 113):
                    print(count)
                    break
                if(count%5 == 0):
                    print(count)
                    