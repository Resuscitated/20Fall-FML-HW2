from svmutil import *
import numpy as np
import matplotlib.pyplot as plt
arrays = []
'''
f=open('./data/abalone.train.scaled')
line = f.readline()
while line:
    arrays.append(line)
    line = f.readline()
f.close()

arrays = np.array(arrays)
np.random.shuffle(arrays)

f = open('./data/abalone.train.scaled.random', 'w')
for i in range(3133):
    f.write(arrays[i])
f.close()
'''
y, x = svm_read_problem('./data/abalone.train.scaled.random')

start_k = -10
end_k = 10
fold = 10
size = len(y)
one_size = int(size/fold)
degree = range(1, 5)
min_mean = 1
min_d = 0
min_k = 0

for d in degree:
    stat_mean=np.ndarray([end_k-start_k+1])
    stat_std=np.ndarray([end_k-start_k+1])
    for k in range(start_k, end_k + 1):
        c_param = 2 ** k
        model_param = '-t 1 -q -d ' + str(d) + ' -c ' + str(c_param)
        err = np.ndarray([fold])
        for i in range(fold):
            y_train = np.append(y[:i * one_size], y[(i + 1) * one_size:])
            x_train = x[:i * one_size] + x[(i + 1) * one_size:]
            model = svm_train(y_train, x_train, model_param)
            y_test = y[i*one_size:(i+1)*one_size]
            x_test = x[i*one_size:(i+1)*one_size]
            p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
            err[i] = 1 - p_acc[0] / 100

        mean = err.mean()
        std = np.std(err)
        stat_mean[k - start_k] = mean
        stat_std[k - start_k] = std

    x_axis = range(start_k, end_k + 1)
    plt.plot(x_axis, stat_mean)
    plt.plot(x_axis, stat_mean + stat_std)
    plt.plot(x_axis, stat_mean - stat_std)
    plt.xticks(range(start_k, end_k + 1, 1))
    plt.legend(['mean','mean + std','mean - std'])
    plt.xlabel('log_2 C')
    plt.ylabel('Cross-validation Error')
    plt.title('d='+str(d))
    plt.show()

    min_temp = np.min(stat_mean)
    if min_temp < min_mean:
        min_mean = min_temp
        min_d = d
        min_k = np.argmin(stat_mean) + start_k

print('best k is ' + str(min_k))
print('best d is ' + str(min_d))
print('best error is ' + str(min_mean))
