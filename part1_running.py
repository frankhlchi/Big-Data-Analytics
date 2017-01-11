__author__ = 'chihongliang'
import cif
import csv
from PIL import Image
#the import of confusion matrix is used for testing
import confusionMatrix
import imageread
from model import EigenModel
import numpy as np

'''
PART one classification
this python script is to classify query image and output them ino CSV file
'''
#load training data
X=cif.load()
y=cif.loadlabel()

#only for writer to test
#xtest = cif.loadtest()[:1000]
#ytest = cif.loadtestlabel()[:1000]

num=imageread.printPath(1,'./COMP3406_assignment1_part1_query')
query = []
#read query image from folder
if num <=10 :
    for i in range(num):
        pic =list(Image.open('./COMP3406_assignment1_part1_query'+'/'+'img'+'0'+str(i)+'.png').getdata())
        query.append(pic)
elif num >10:
    for i in range(10):
        pic=list(Image.open('./COMP3406_assignment1_part1_query'+'/'+'img'+'0'+str(i)+'.png').getdata())
        pic = [x for sets in pic for x in sets]
        query.append(pic)
    for i in range(10,num):
        pic=list(Image.open('./COMP3406_assignment1_part1_query'+'/'+'img'+str(i)+'.png').getdata())
        pic = [x for sets in pic for x in sets]
        query.append(pic)

#set number of eigenvalues equal to 50
model = EigenModel(X,y,None,50)
predict =[]

for i in range(num):
    predict.append(model.predict(np.array(query[i])))

#write the csv file
csvfile = file('part1_cl.csv','wb')
writer=csv.writer(csvfile)
writer.writerow(predict)
csvfile.close()
