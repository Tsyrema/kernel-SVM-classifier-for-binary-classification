#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from matplotlib.colors import ListedColormap
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# acquire data, split it into training and testing sets (50% each)
# nc -- number of classes for synthetic datasets
def acquire_data(data_name, nc = 2):
    if data_name == 'synthetic-easy':
        print 'Creating easy synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 0 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-medium':
        print 'Creating medium synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 3 * rng.uniform(size=X.shape)
    elif data_name == 'synthetic-hard':
        print 'Creating hard easy synthetic labeled dataset'
        X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes = nc, random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 5 * rng.uniform(size=X.shape)
    elif data_name == 'moons':
        print 'Creating two moons dataset'
        X, y = datasets.make_moons(noise=0.2, random_state=0)
    elif data_name == 'circles':
        print 'Creating two circles dataset'
        X, y = datasets.make_circles(noise=0.2, factor=0.5, random_state=1)
    elif data_name == 'iris':
        print 'Loading iris dataset'
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif data_name == 'digits':
        print 'Loading digits dataset'
        digits = datasets.load_digits()
        X = digits.data
        y = digits.target
    elif data_name == 'breast_cancer':
        print 'Loading breast cancer dataset'
        bcancer = datasets.load_breast_cancer()
        X = bcancer.data
        y = bcancer.target
    else:
        print 'Cannot find the requested data_name'
        assert False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test

# compare the prediction with grount-truth, evaluate the score
def myscore(y, y_gt):
    assert len(y) ==  len(y_gt)
    return np.sum(y == y_gt)/float(len(y))

# plot data on 2D plane
# use it for debugging
def draw_data(X_train, X_test, y_train, y_test, nclasses):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    cm = plt.cm.jet
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, edgecolors='k', marker='x', linewidth = 3, label='Test Data')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()

# draw results on 2D plan for binary classification
# this is a fake version (using a random linear classifier)
# modify it for your own usage (pass in parameter etc)
def draw_result_binary_fake(X_train, X_test, y_train, y_test):

    h = .02
    X = np.vstack([X_train, X_test])
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Put the result into a color plot
    tmpX = np.c_[xx.ravel(), yy.ravel()]

    Z_class, Z_pred_val = get_prediction_fake(tmpX)

    Z_clapped = np.zeros(Z_pred_val.shape)
    Z_clapped[Z_pred_val>=0] = 1.5
    Z_clapped[Z_pred_val>=1.0] = 2.0
    Z_clapped[Z_pred_val<0] = -1.5
    Z_clapped[Z_pred_val<-1.0] = -2.0

    Z = Z_clapped.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdBu, alpha = .4)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    #    ax = plt.figure(1)
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k', label='Training Data')
    # and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', marker='x', linewidth=3,
                label='Test Data')

    y_train_pred_class, y_train_pred_val = get_prediction_fake(X_train)
    sv_list_bool = np.logical_and(y_train_pred_val >= -1.0, y_train_pred_val <= 1.0)
    sv_list = np.where(sv_list_bool)[0]
    plt.scatter(X_train[sv_list, 0], X_train[sv_list, 1], s=100, facecolors='none', edgecolors='orange', linewidths = 3, label='Support Vectors')

    y_test_pred_class, y_test_pred_val = get_prediction_fake(X_test)
    score = myscore(y_test_pred_class, y_test)
    plt.text(xx.max() - .3, yy.min() + .3, ('Score = %.2f' % score).lstrip('0'), size=15, horizontalalignment='right')

    plt.legend()
    plt.show()

# predict labels using a random linear classifier
# returns a list of length N, each entry is either 0 or 1
def get_prediction_fake(X):
    np.random.seed(100)
    nfeatures = X.shape[1]
    # w = np.random.rand(nfeatures + 1) * 2.0
    w = [-1,0,0]

    assert len(w) == X.shape[1] + 1
    w_vec = np.reshape(w,(-1,1))
    X_extended = np.hstack([X, np.ones([X.shape[0],1])])
    y_pred_value = np.ravel(np.dot(X_extended,w_vec))
    y_pred_class = np.maximum(np.zeros(y_pred_value.shape), y_pred_value)
    return y_pred_class, y_pred_value
    print 'Finished. Took:', time.time() - startTime


####################################################
# binary label classification

def linear_kernel (X, y, kpar):
    return np.dot(X,y)

def poly_kernel(X, y, kpar):
    return (1 + np.dot(X,y))**kpar

def gaussian_kernel (X, y, kpar):
    y = np.transpose(y)
    e = np.exp(-np.linalg.norm(X-y)**2 / (2*kpar))
    return e


def kernel_matrix(X, y, ker, kpar):
    num1 = X.shape[0]
    num2 = y.shape[0]
    K = np.zeros((num1, num2))
    for i in range(num1):
        for j in range(num2):
            if ker == 'linear':
                K[i,j] = linear_kernel(X[i], y[j], kpar)
            if ker == 'polynomial':
                K[i,j] = poly_kernel(X[i], y[j], kpar)
            if ker == 'gaussian':
                K[i,j] = gaussian_kernel(X[i], y[j], kpar)
    return K

#change sign of y_train
def changeSign(s):
    if s <= 0.0:
        return -1
    else:
        return 1


# training kernel svm
# return sv_list: list of surport vector IDs
# alpha: alpha_i's
# b: the bias
def mytrain_binary(X_train, y_train, C, ker, kpar):
    print 'Start training ...'
    # calculate training time
    startTime = time.time()

    num = X_train.shape[0]

    K = kernel_matrix(X_train, X_train, ker, kpar)
    # 1/2 X^T*P*X + q^T*X
    # GX<=h
    # AX=b

    P = cvxopt.matrix(np.outer(y_train,y_train) * K)
    q = cvxopt.matrix(np.ones(num)* -1)

    y_train = y_train.astype(np.double)
    A = cvxopt.matrix(y_train, (1, num))

    B = cvxopt.matrix(0.0)

    # a_i <=0
    if C is None:
        # G = cvxopt.matrix(-np.eye(num))
        G = cvxopt.matrix(np.diag(np.ones(num)* -1))
        h = cvxopt.matrix(np.zeros(num))
    # a_i <=C
    else:
        m1 = np.diag(np.ones(num)* -1)
        m2 = np.identity(num)
        G = cvxopt.matrix(np.vstack((m1,m2)))
        m1 = np.zeros(num)
        m2 = np.ones(num)* C
        h = cvxopt.matrix(np.hstack((m1,m2)))

    sol = cvxopt.solvers.qp(P, q, G, h, A, B)
    alpha = np.ravel(sol['x'])

    sv = np.nonzero(alpha > 1.0e-3)   #indices
    sv_list = np.ravel(sv)
    alpha_support = alpha[sv_list]  #alphas >0

    sv_vectors = X_train[sv_list]   #support vectors
    sv_labels = y_train[sv_list]    # support vectors labels
    # print 'sv_labels: ', sv_labels

    ker_values = kernel_matrix(sv_vectors,sv_vectors, ker, kpar)
    # print 'kerval: ', ker_values

    b = np.average(sv_labels - np.dot(np.transpose(alpha_support * sv_labels), ker_values))

    # print sv_list
    # print 'alpha: ' , alpha
    # print 'b: ', b
    print 'Finished training. Took:', time.time() - startTime
    return sv_list, alpha, b

# predict given X_test data,
# need to use X_train, ker, kpar_opt to compute kernels
# need to use sv_list, y_train, alpha, b to make prediction
# return y_pred_class as classes (convert to 0/1 for evaluation)
# return y_pred_value as the prediction score
def mytest_binary(X_test, X_train, y_train, sv_list, alpha, b, ker, kpar):
    num = X_test.shape[0]
    alphas = alpha[sv_list]
    alphas = np.transpose(np.reshape(alphas, (-1,1)))
    sv_vectors = X_train[sv_list]  # support vectors
    # print 'X_train: ', sv_vectors

    sv_labels = y_train[sv_list]  # suppor vectors labels
    sv_labels = np.transpose(np.reshape(sv_labels, (-1,1)))

    K = kernel_matrix(sv_vectors, X_test, ker, kpar)

    y_pred_value = np.dot((alphas * sv_labels), K) + b
    y_pred_value = np.ravel(y_pred_value)

    y_sign = np.vectorize(changeSign)
    y_pred_class = np.ravel(y_sign(y_pred_value))

    # print 'y_pred_class: ', y_pred_class
    return y_pred_class, y_pred_value


# split X_train, y_train to new X_train and y_yrain according to a current fold i
# k is the number of folds
def split (X, y, i, k):
    size = X.shape[0]
    fold_size = size / k
    start = i * fold_size
    end = start + fold_size
    X_test = X[start:end]
    y_test = y[start:end]
    X_train = np.concatenate((X[:start], X[end:]))
    y_train = np.concatenate((y[:start], y[end:]))
    return X_test, y_test, X_train, y_train



# use cross validation to decide the optimal C and the kernel parameter kpar
# if linear, kpar = -1 (no meaning)
# if polynomial, kpar is the degree
# if gaussian, kpar is sigma-square
# k -- number of folds for cross-validation, default value = 5
def my_cross_validation(X_train, y_train, ker, k = 5):
    assert ker == 'linear' or ker == 'polynomial' or ker == 'gaussian'

    parameters_score = []
    fold_score = []
    for C in range(1, 5, 1):
        for kpar in range(1, 5, 1):
            for fold in range (k):
                X_test_new, y_test_new, X_train_new, y_train_new = split(X_train, y_train, fold, k)
                sv_list, alpha ,b = mytrain_binary(X_train_new, y_train_new, C, ker, kpar)
                y_pred_class, y_pred_value = mytest_binary(X_test_new, X_train_new, y_train_new, sv_list, alpha, b, ker, kpar)
                test_score = myscore(y_pred_class, y_test_new)
                fold_score.append(test_score)
            average_score = np.average(fold_score)
            print 'average score: ', average_score
            parameters_score.append((C, kpar, average_score))
    C_opt = 0.0
    kpar_opt = 0.0
    max_score = 0.0
    for tup in parameters_score:
        if tup[2] > max_score:
            max_score = tup[2]
            C_opt = tup[0]
            kpar_opt = tup[1]
    if ker == 'linear':
        kpar_opt = -1 #dummy
    print 'C_opt: ', C_opt
    print 'kpar_opt: ', kpar_opt
    return C_opt, kpar_opt




################

def main():

    #######################
    # get data
    # only use binary labeled

    X_train, X_test, y_train, y_test = acquire_data('synthetic-easy')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-medium')
    # X_train, X_test, y_train, y_test = acquire_data('synthetic-hard')
    # X_train, X_test, y_train, y_test = acquire_data('moons')
    # X_train, X_test, y_train, y_test = acquire_data('circles')
    # X_train, X_test, y_train, y_test = acquire_data('breast_cancer')

    y_sign = np.vectorize(changeSign)
    y_train = y_sign(y_train)
    y_test = y_sign(y_test)

    nfeatures = X_train.shape[1]    # number of features
    ntrain = X_train.shape[0]   # number of training data
    ntest = X_test.shape[0]     # number of test data
    y = np.append(y_train, y_test)
    nclasses = len(np.unique(y)) # number of classes

    # only draw data (on the first two dimension)
    # draw_data(X_train, X_test, y_train, y_test, nclasses)
    # a face function to draw svm results
    # draw_result_binary_fake(X_train, X_test, y_train, y_test)

    ker = 'linear'
    # ker = 'polynomial'
    # ker = 'gaussian'

    C_opt, kpar_opt = my_cross_validation(X_train, y_train, ker, 5)
    sv_list, alpha, b = mytrain_binary(X_train, y_train, C_opt, ker, kpar_opt)

    y_test_pred_class, y_test_pred_val = mytest_binary(X_test, X_train, y_train, sv_list, alpha, b, ker, kpar_opt)

    test_score = myscore(y_test_pred_class, y_test)

    print 'Test Score:', test_score

if __name__ == "__main__": main()
