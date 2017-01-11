__author__ = 'chihongliang'
import os
allFileNum = 0
#it is to read the num of images in query folder

def printPath(level, path):
    global allFileNum
    dirList = []
    fileList = []
    files = os.listdir(path)
    dirList.append(str(level))
    for f in files:
        if(os.path.isdir(path + '/' + f)):
            if(f[0] == '.'):
                pass
            else:
                dirList.append(f)
        if(os.path.isfile(path + '/' + f)):
            fileList.append(f)
    for fl in fileList:
        allFileNum = allFileNum + 1
    return allFileNum-1
