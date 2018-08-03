from os import walk
import os
import cv2
import numpy as np

f = []
mypath = "/home/alpha/MLProject/c3d_feature_extraction/Input"
for (dirpath, dirnames, filenames) in walk(mypath):
    for (dirpath, dirnames, filenames) in walk(dirpath):
        f.extend(dirpath for x in filenames)
        break
print(len(f))

f = set(f)
f = list(f)


def getSettingFiles(list):
    input_frm = open('input_list_frm.txt', 'w')
    output_frm = open('output_list_prefix.txt', 'w')
    for fl in list:       
        num = len(os.listdir(fl))
        j = int(num/16)
        temp = fl.split('/')
        temp[-3] = 'FInput'
        file_pre = '/'.join(temp[-3:])
        if not os.path.exists(file_pre):
            os.makedirs(file_pre)
        bla = fl.split('/')
        ula = '/'.join(bla[-3:])
        for i in range (0,j):
            pf = str(1+ i*16) + " 0"
            name = ula + "/ " + pf
            input_frm.write("%s\n" % name)
            out = file_pre + "/" + str(1+i*16).zfill(6)
            output_frm.write('%s\n' % out)
getSettingFiles(f)
