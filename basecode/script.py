import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    y_classes = np.unique(y)

    dimension = (len(y_classes), X.shape[1])
    means = np.zeros(dimension)

    counter = 0
    for i in y_classes:
        split_X = X[np.where(y == i)[0]]
        means[counter] = np.mean(split_X, axis=0)
        counter += 1

    means = means.T
    covmat = np.cov(X, rowvar=False, bias=1)
    return means, covmat


def qdaLearn(X, y):
    y_classes = np.unique(y)

    dimension = (len(y_classes), X.shape[1])
    means = np.zeros(dimension)
    covmats = []

    counter = 0
    for i in y_classes:
        split_X = X[np.where(y == i)[0]]
        means[counter] = np.mean(split_X, axis=0)
        covmats.append(np.cov(split_X, rowvar=False, bias=1))
        counter += 1

    means = means.T
    covmats = np.array(covmats)
    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    sigma = det(covmat)

    pdf = np.zeros((Xtest.shape[0], means.shape[1]))

    for i in range(means.shape[1]):
        term1 = Xtest - means[:, i]
        pdf[:, i] = np.exp(-0.5 * np.sum(term1 * np.dot(inv(covmat), term1.T).T, 1)) / (np.sqrt(2 * np.pi * sigma))

    ypred = (np.argmax(pdf, 1)) + 1
    ypred = ypred.reshape(-1, 1)
    acc = 0;
    for i in range(Xtest.shape[0]):
        if ypred[i] == ytest[i]:
            acc = acc + 1;
    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    pdf = np.zeros((Xtest.shape[0], means.shape[1]))

    for i in range(means.shape[1]):
        term1 = Xtest - means[:, i]
        pdf[:, i] = np.exp(-0.5 * np.sum(term1 * np.dot(inv(covmats[i]), term1.T).T, 1)) / (
            np.sqrt(2 * np.pi * det(covmats[i])))

    ypred = (np.argmax(pdf, 1)) + 1
    ypred = ypred.reshape(-1, 1)
    acc = 0;
    for i in range(Xtest.shape[0]):
        if ypred[i] == ytest[i]:
            acc = acc + 1;
    return acc, ypred


def learnOLERegression(X, y):
    term1 = inv(np.dot(X.T, X))
    term2 = np.dot(X.T, y)
    w = np.dot(term1, term2)
    return w


def learnRidgeRegression(X, y, lambd):
    d = np.shape(X)[1]
    term1 = np.dot(X.T, X) + np.dot(lambd, np.identity(d))
    term2 = np.dot(X.T, y)

    w = np.dot(inv(term1), term2)

    return w


def testOLERegression(w, Xtest, ytest):
    N = Xtest.shape[0]
    term = ytest - np.dot(Xtest, w)
    mse = (np.dot(term.T, term)) / N
    return mse


def regressionObjVal(w, X, y, lambd):
    term11 = np.dot(w.T, X.T)
    term1 = y.T - term11
    term1_sq = np.square(term1)
    flattened_term1_sq = np.sum(term1_sq)
    final_term1 = flattened_term1_sq / 2

    term21 = np.dot(w.T, w)
    term2 = np.dot(lambd, term21)
    final_term2 = term2 / 2
    error = final_term1 + final_term2

    grad_term1 = np.dot(lambd, w)
    grad_term2 = np.dot(y.T, X)
    grad_term31 = np.dot(X.T, X)
    grad_term3 = np.dot(w.T, grad_term31)
    error_grad = grad_term1 - grad_term2 + grad_term3

    error = error.flatten()
    error_grad = error_grad.flatten()

    return error, error_grad


def mapNonLinear(x, p):
    Xp = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xp[:, i] = x ** i
    return Xp


# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# LDA
means, covmat = ldaLearn(X, y)
ldaacc, ldares = ldaTest(means, covmat, Xtest, ytest)
print('LDA Accuracy = ' + str(ldaacc))
# QDA
means, covmats = qdaLearn(X, y)
qdaacc, qdares = qdaTest(means, covmats, Xtest, ytest)
print('QDA Accuracy = ' + str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)

zacc, zldares = ldaTest(means, covmat, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zldares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc, zqdares = qdaTest(means, covmats, xx, np.zeros((xx.shape[0], 1)))
plt.contourf(x1, x2, zqdares.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w = learnOLERegression(X, y)
mle = testOLERegression(w, Xtest, ytest)

w_i = learnOLERegression(X_i, y)
mle_i = testOLERegression(w_i, Xtest_i, ytest)

print('MSE without intercept ' + str(mle))
print('MSE with intercept ' + str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1

opt = lambdas[np.argmin(mses3)]
# print(lambdas[np.argmin(mses3_train)])
# print(mses3_train)

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k, 1))
mses4 = np.zeros((k, 1))
opts = {'maxiter': 20}  # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args, method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l, [len(w_l), 1])
    mses4_train[i] = testOLERegression(w_l, X_i, y)
    mses4[i] = testOLERegression(w_l, Xtest_i, ytest)
    i = i + 1
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses4_train)
plt.plot(lambdas, mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses4)
plt.plot(lambdas, mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize', 'Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = opt  # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax, 2))
mses5 = np.zeros((pmax, 2))
for p in range(pmax):
    Xd = mapNonLinear(X[:, 2], p)
    Xdtest = mapNonLinear(Xtest[:, 2], p)
    w_d1 = learnRidgeRegression(Xd, y, 0)
    mses5_train[p, 0] = testOLERegression(w_d1, Xd, y)
    mses5[p, 0] = testOLERegression(w_d1, Xdtest, ytest)
    w_d2 = learnRidgeRegression(Xd, y, lambda_opt)
    mses5_train[p, 1] = testOLERegression(w_d2, Xd, y)
    mses5[p, 1] = testOLERegression(w_d2, Xdtest, ytest)

fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax), mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization', 'Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax), mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization', 'Regularization'))
plt.show()
