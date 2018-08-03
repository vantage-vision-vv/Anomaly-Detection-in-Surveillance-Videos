import os
from os import walk
import numpy as np
from sklearn.preprocessing import normalize

mypath = "/home/alpha/MLProject/Feature Extraction/FInput"
output = "/home/alpha/MLProject/Feature Extraction/FOutput/"
for (dirpath, dirnames, filenames) in walk(mypath):
    for (dirp, dirn, fn) in walk(dirpath):
        key = []
        value = []
        if len(filenames) == 0:
                continue
        for fl in filenames:
            norm = normalize(np.array(map(float,open(os.path.join(dirpath, fl)).readlines()[0].strip().split(','))).reshape(1,-1),norm='l2')
            print(norm.shape)
            value.append(norm[0].tolist())
            key.append(int(fl.split('.')[0]))
        final_vector = []
        value = [x for _, x in sorted(zip(key, value))]
        key = sorted(key)
        width = int(key[-1])/32
        res = [x/width for x in key]
        start = 0
        a = res[0]
        for i in range(len(key)):
            if a != res[i]:
                final_vector.append(np.mean(np.array(value[start:i]),axis=0).tolist())
                a = res[i]
                start = i
        if not os.path.exists(output + dirpath.split("/")[-2]):
            os.makedirs(output + dirpath.split("/")[-2])
        with open(output + dirpath.split("/")[-2]+"/"+ dirpath.split("/")[-1]+".csv",'w') as f:
            f.write(str(width)+"\n")
            for vec in final_vector:
                rec = ""
                for item in vec:
                    rec = rec + str(item) + ","
                rec = rec[:-1]
                rec = rec + "\n"
                f.write(rec)


                
