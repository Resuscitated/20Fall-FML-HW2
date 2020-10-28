from svmutil import *
import numpy as np
import matplotlib.pyplot as plt

y, x = svm_read_problem('./data/abalone.train.scaled.random')
y_pred, x_pred = svm_read_problem('./data/abalone.test.scaled')

k = 10
c_param = 2 ** k
fold = 10
size = len(y)
one_size = int(size/fold)
degree = range(1, 5)
cv_err=np.ndarray([4])
test_err=np.ndarray([4])
sv_number = np.ndarray([4])
margin_sv_number = np.ndarray([4])
for d in degree:
    model_param = '-t 1 -q -d ' + str(d) + ' -c ' + str(c_param)
    cv = np.ndarray([fold])
    test = np.ndarray([fold])
    sv = np.ndarray([fold])
    margin_sv = np.ndarray([fold])
    for i in range(fold):
        y_train = np.append(y[:i * one_size], y[(i + 1) * one_size:])
        x_train = x[:i * one_size] + x[(i + 1) * one_size:]
        model = svm_train(y_train, x_train, model_param)
        y_test = y[i * one_size:(i + 1) * one_size]
        x_test = x[i * one_size:(i + 1) * one_size]
        sv[i] = model.get_nr_sv()
        index = model.get_sv_indices()
        coef = model.get_sv_coef()
        sv_y = [y_train[j - 1] for j in index]
        margin_sv[i] = 0
        for j in range(int(sv[i])):
            alpha = sv_y[j] * coef[j][0]
            if alpha != c_param:
                margin_sv[i] += 1
        p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
        cv[i] = 1 - p_acc[0] / 100
        p_label, p_acc, p_val = svm_predict(y_pred, x_pred, model)
        test[i] = 1 - p_acc[0]/100

    cv_err[d-1] = cv.mean()
    test_err[d-1] = test.mean()
    sv_number[d-1] = sv.mean()
    margin_sv_number[d-1] = margin_sv.mean()

x_axis = degree
plt.plot(x_axis, cv_err)
plt.plot(x_axis, test_err)
plt.xticks(range(1, 5, 1))
plt.legend(['Cross-validation Error', 'Test Error'])
plt.xlabel('degree d')
plt.ylabel('error')
plt.show()

x_axis = degree
plt.plot(x_axis, sv_number)
plt.xticks(range(1, 5, 1))
plt.title('Number of Support Vectors')
plt.legend()
plt.xlabel('degree d')
plt.ylabel('Count')
plt.show()

x_axis = degree
plt.plot(x_axis, margin_sv_number)
plt.xticks(range(1, 5, 1))
plt.title('Number of Marginal Support Vectors')
plt.xlabel('degree d')
plt.ylabel('Count')
plt.show()
