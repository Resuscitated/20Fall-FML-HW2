from svmutil import *
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import os

'''
gamma = 1 / 10
coef0 = 0

arrays_train = []
arrays_test = []
f = open("./data/abalone.train")
line = f.readline()
while line:
    arrays_train.append(line)
    line = f.readline()
f.close()

f = open("./data/abalone.test")
line = f.readline()
while line:
    arrays_test.append(line)
    line = f.readline()
f.close()

train_data = np.ndarray([3133,11])
test_data = np.ndarray([1044,11])

for i in range(3133):
    temp = arrays_train[i].split(' ')
    train_data[i, 0] = float(temp[0])
    for j in range(1,11):
        a = temp[j].split(':')
        train_data[i, j]= float(a[1])

for i in range(1044):
    temp = arrays_test[i].split(' ')
    test_data[i, 0] = float(temp[0])
    for j in range(1, 11):
        a = temp[j].split(':')
        test_data[i, j] = float(a[1])


train_data_label= train_data[:,0]
train_data_attr=train_data[:,1:]

test_data_label= train_data[:,0]
test_data_attr=train_data[:,1:]

for d in range(1, 5):
    new_train_data_attr = np.power(np.dot(train_data_attr,np.transpose(train_data_attr))*gamma + coef0, d)
    new_train_data_attr = np.dot(new_train_data_attr, np.diag(train_data_label))
    new_test_data_attr = np.power(np.dot(test_data_attr,np.transpose(train_data_attr))*gamma + coef0, d)
    new_test_data_attr =  np.dot(new_test_data_attr, np.diag(train_data_label))
    new_train_data= np.insert(new_train_data_attr,0,train_data_label,axis=1)
    new_test_data= np.insert(new_test_data_attr,0,test_data_label,axis=1)

    new_train_data = new_train_data.tolist()
    new_test_data = new_test_data.tolist()

    train = open('./data/abalone.train_'+str(d), 'w')
    for i in range(0, 3133):
        st = str(new_train_data[i][0])
        for j in range(1, 3134):
            st = st + ' ' + str(j) + ':' + str(new_train_data[i][j])
        if i != 3132:
            train.write(st + "\n")
        else:
            train.write(st)
    train.close()

    test = open('./data/abalone.test_' + str(d), 'w')
    for i in range(3133-3133, 4177-3133):
        st = str(new_test_data[i][0])
        for j in range(1, 3134):
            st = st + ' ' + str(j) + ':' + str(new_test_data[i][j])
        if i != 4176-3133:
            test.write(st + "\n")
        else:
            test.write(st)
    test.close()


for d in range(1,5):
    os.system(".\data\svm-scale -s ./data/range_"+str(d)+" ./data/abalone.train_"+str(d)+" > ./data/abalone.train.scaled_"+str(d))
    os.system(".\data\svm-scale -r ./data/range_"+str(d)+" ./data/abalone.test_"+str(d)+" > ./data/abalone.test.scaled_"+str(d))
'''

start_k = -10
end_k = 10
fold = 10
one_size = 313
degree = range(1, 5)
min_mean = 1
min_d = 0
min_k = 0
y = np.ndarray([4,3133])
x = []
scale_param= []
for d in degree:
    y[d - 1, :], temp_x = svm_read_problem('./data/abalone.train.scaled_' + str(d), return_scipy=True)
    x.append(temp_x)

    '''
    # y[d - 1, :], temp_x= svm_read_problem('./data/abalone.train_'+str(d), return_scipy=True)
    # x.append(temp_x)
    # scale_param.append(csr_find_scale_param(x[d-1]))
    # x[d-1] = csr_scale(x[d-1], scale_param[d-1])

    stat_mean = np.ndarray([end_k - start_k + 1])
    stat_std = np.ndarray([end_k - start_k + 1])
    for k in range(start_k, end_k + 1):
        c_param = 2 ** k
        model_param = '-q -c ' + str(c_param)
        err = np.ndarray([fold])
        for i in range(fold):
            y_train = np.append(y[d-1][:i * one_size], y[d-1][(i + 1) * one_size:])
            x_train = scipy.sparse.vstack([x[d-1][:i * one_size], x[d-1][(i + 1) * one_size:]])
            model = svm_train(y_train, x_train, model_param)
            y_test = y[d-1][i * one_size:(i + 1) * one_size]
            x_test = x[d-1][i * one_size:(i + 1) * one_size]
            p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
            err[i] = 1 - p_acc[0] / 100

        mean = err.mean()
        std = np.std(err)
        stat_mean[k - start_k] = mean
        stat_std[k - start_k] = std

    min_temp = np.min(stat_mean)
    if min_temp < min_mean:
        min_mean = min_temp
        min_d = d
        min_k = np.argmin(stat_mean) + start_k

print('best k is ' + str(min_k))
print('best d is ' + str(min_d))

'''

min_k = 10
y_pred = np.ndarray([4, 1044])
x_pred = []
cv_err = np.ndarray([4])
test_err = np.ndarray([4])
c_param = 2 ** min_k
for d in degree:
    
    # y_pred[d - 1, :], temp_x = svm_read_problem('./data/abalone.test_' + str(d), return_scipy=True)
    # x_pred.append(temp_x)
    # x_pred[d-1]=csr_scale(x_pred[d-1], scale_param[d-1])
    
    y_pred[d - 1, :], temp_x = svm_read_problem('./data/abalone.test.scaled_' + str(d), return_scipy=True)
    x_pred.append(temp_x)
    model_param = '-q -c ' + str(c_param)
    cv = np.ndarray([fold])
    test = np.ndarray([fold])
    for i in range(fold):
        y_train = np.append(y[d - 1][:i * one_size], y[d - 1][(i + 1) * one_size:])
        x_train = scipy.sparse.vstack([x[d - 1][:i * one_size], x[d - 1][(i + 1) * one_size:]])
        model = svm_train(y_train, x_train, model_param)
        y_test = y[d - 1][i * one_size:(i + 1) * one_size]
        x_test = x[d - 1][i * one_size:(i + 1) * one_size]
        p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
        cv[i] = 1 - p_acc[0] / 100
        p_label, p_acc, p_val = svm_predict(y_pred[d-1], x_pred[d-1], model)
        test[i] = 1 - p_acc[0] / 100
    cv_err[d - 1] = cv.mean()
    test_err[d - 1] = test.mean()

x_axis = degree
plt.plot(x_axis, cv_err)
plt.plot(x_axis, test_err)
plt.xticks(range(1, 5, 1))
plt.legend(['Cross-validation Error', 'Test Error'])
plt.xlabel('degree d')
plt.ylabel('Error')
plt.show()
