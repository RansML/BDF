import numpy as np
import csv
import sys
import scipy as sp
import os
from sklearn import metrics
from datetime import datetime

def read_txy_csv(fn):
   data = readCsvFile(fn)
   Xtest = data[:, :3]
   Ytest = data[:, 3][:, np.newaxis]
   return Xtest, Ytest


def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    return np.array( dataList )

def entropy(x):
    x[x < sys.max_info.min] = sys.max_info.min
    return -1*np.sum(x*np.log2(x))

def cross_entropy(act, pred):
    #negative log-loss sklearn
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred))
    return -ll

def mean_standardized_log_loss(true_labels, predicted_mean, predicted_var):
    """
    :param true_labels:
    :param predicted_mean:
    :param predicted_var:
    :return: 0 for simple methods, negative for better methods (eq. 2.34 GPML book)
    """
    predicted_var = predicted_var + 10e6*np.finfo(float).eps #to avoid /0 and log(0)
    msll = np.average(0.5*np.log(2*np.pi*predicted_var) + ((predicted_mean - true_labels)**2)/(2*predicted_var))
    msll *= -1
    return msll

def calc_scores_velocity(mdl_name, query_type, true, predicted, predicted_var=None, train_time=-1, query_time=-1, save_report=True, notes=''):
    fn = mdl_name+ '.csv'

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    msll = mean_standardized_log_loss(true, predicted, predicted_var)

    print(' Metrics: RMSE={:.3f}, MSLL={:.3f}, train_time={:.3f}, query_time={:.3f}'.format(rmse, msll, train_time, query_time))
    if save_report is True:
       if os.path.isfile(fn): # If the file already exists
           header = ''
       else:
           header = 'Time, Tested on, RMSE, MSLL, Train time, Query time, Notes'
       with open(fn,'ab') as f_handle: #try 'a'
          np.savetxt(f_handle, np.array([[datetime.now(), query_type, rmse, msll, train_time, query_time, notes]]), \
                     delimiter=',', fmt='%s, %s, %.3f, %.3f, %.3f, %.3f, %s', header=header, comments='')

def calc_scores_occupancy(mdl_name, true, predicted, predicted_var=None, time_taken=-11, N_points=0, do_return=False,
                         save_report=True):
    #TODO: double check
    fn = mdl_name + '.csv'

    predicted_binarized = np.int_(predicted >= 0.5)
    accuracy = np.round(metrics.accuracy_score(true.ravel(), predicted_binarized.ravel()), 3)

    auc = np.round(metrics.roc_auc_score(true.ravel(), predicted.ravel()), 3)

    nll = np.round(metrics.log_loss(true.ravel(), predicted.ravel()), 3)

    if predicted_var is not None:
        neg_smse = np.round(neg_ms_log_loss(true, predicted[0].ravel(), predicted_var.ravel()), 3)
    else:
        neg_smse = -11

    print(mdl_name + ': accuracy={}, auc={}, nll={}, smse={}, time_taken={}'.format(accuracy, auc, nll, neg_smse,
                                                                                    time_taken))
    # print(metrics.confusion_matrix(true.ravel(), predicted_binarized.ravel()))
    if save_report is True:
        with open(fn, 'ab') as f_handle:  # try 'a'
            # np.savetxt(f_handle, np.array([[neg_smse]]), delimiter=',', fmt="%.3f")
            np.savetxt(f_handle, np.array([[accuracy, auc, nll, neg_smse, time_taken, N_points]]), delimiter=',',
                       fmt="%.3f")
    if do_return:
        return accuracy, auc, nll
