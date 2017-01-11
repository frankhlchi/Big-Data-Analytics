__author__ = 'chihongliang'
import cif100
import numpy as np
from model import EigenModel
import imageread
import confusionMatrix
#the import of confusion matrix is used for testing
from PIL import Image
import csv

'''
PART two classification
this python script is to classify query image and generate inference on class and supclass in CSV file
'''

#read query image from folder
num=imageread.printPath(1,'./COMP3406_assignment1_part2_query')
query = []

if num <=10 :
    for i in range(num):
        pic =list(Image.open('./COMP3406_assignment1_part2_query'+'/'+'img'+'0'+str(i)+'.png').getdata())
        query.append(pic)
elif num >10:
    for i in range(10):
        pic=list(Image.open('./COMP3406_assignment1_part2_query'+'/'+'img'+'0'+str(i)+'.png').getdata())
        pic = [x for sets in pic for x in sets]
        query.append(pic)
    for i in range(10,num):
        pic=list(Image.open('./COMP3406_assignment1_part2_query'+'/'+'img'+str(i)+'.png').getdata())
        pic = [x for sets in pic for x in sets]
        query.append(pic)

[X,y,z] = [cif100.traindata(),cif100.traincoarselabel(),cif100.trainfinelabel()]
#xtest = cif100.testdata()
#ytest = cif100.testcoarselabel()
model = EigenModel(X,y,z,100)

predict =[]
predicttwo =[]
for i in range(num):
    prediction = model.predicttwo(np.array(query[i]))
    #print prediction
    predict.append(prediction[0])
    predicttwo.append(prediction[1])


#write the CSV file
csvfile = file('part1_supclass.csv','wb')
writer=csv.writer(csvfile)
writer.writerow(predict)
csvfile.close()

csvfiletwo = file('part1_class.csv','wb')
writertwo=csv.writer(csvfiletwo)
writertwo.writerow(predicttwo)
csvfile.close()