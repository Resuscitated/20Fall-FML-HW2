import numpy as np
import os

arrays = []

f = open("./data/abalone.data")
line = f.readline()
while line:
    data = line.split(',')
    if data[0] == 'M':
        data[0] = '1'
        data.insert(1, '0')
        data.insert(2, '0')
    if data[0] == 'F':
        data[0] = '0'
        data.insert(1, '1')
        data.insert(2, '0')
    if data[0] == 'I':
        data[0] = '0'
        data.insert(1, '0')
        data.insert(2, '1')
    shape = len(data)
    data[shape-1]=data[shape-1][:-1]
    if int(data[shape - 1]) <= 9:
        data[shape - 1] = '-1'
    else:
        data[shape - 1] = '1'
    data.insert(0,data[shape-1])
    data=data[:-1]
    arrays.append(data)
    line = f.readline()
f.close()

train_data = arrays[:3133]
test_data = arrays[3133:]

train = open('./data/abalone.train', 'w')
for i in range(0, 3133):
    for j in range(1, 11):
        train_data[i][j] = str(j) + ':' + train_data[i][j]
    if i != 3132:
        train.write(" ".join(j for j in train_data[i]) + "\n")
    else:
        train.write(" ".join(j for j in train_data[i]))
train.close()

test = open('./data/abalone.test', 'w')
for i in range(3133-3133, 4177-3133):
    for j in range(1, 11):
        test_data[i][j] = str(j) + ':' + test_data[i][j]
    if i != 4176-3133:
        test.write(" ".join(j for j in test_data[i]) + "\n")
    else:
        test.write(" ".join(j for j in test_data[i]))
test.close()

os.system(".\data\svm-scale -s ./data/range ./data/abalone.train > ./data/abalone.train.scaled")
os.system(".\data\svm-scale -r ./data/range ./data/abalone.test > ./data/abalone.test.scaled")