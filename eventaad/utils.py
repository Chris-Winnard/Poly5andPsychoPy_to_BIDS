import os
import sys
from os.path import isfile
from datetime import date
import glob
import random
import matplotlib.pylab as plt
import numpy as np

def one_hot_encode(numOfClasses, labels):
    encoded_labels = np.zeros((numOfClasses, len(labels)), dtype=int)
    for i in range(0, len(labels)):
        if labels[i] > 0:
            encoded_labels[labels[i]-1, i] = 1
            
    return encoded_labels

def getFileList(in_path):
    filepaths = []
    if os.path.isfile(in_path):
        filepaths.append(in_path)
    elif os.path.isdir(in_path):
        for filename in glob.glob(in_path + '/**/*.*', recursive=True):
            filepaths.append(filename)
    else:
        print("Path is invalid: " + in_path)
        return None

    return filepaths

    
def skewed_error_analysis(Y, Y_pred, filepaths, printpath = False):
    true_pos = ((Y==1)&(Y_pred==1))
    false_pos = ((Y==0)&(Y_pred==1))
    true_neg = ((Y==0)&(Y_pred==0))
    false_neg = ((Y==1)&(Y_pred==0))
    
    true_pos_count = np.count_nonzero(true_pos == True, axis=1, keepdims=True)
    false_pos_count = np.count_nonzero(false_pos == True, axis=1, keepdims=True)
    true_neg_count = np.count_nonzero(true_neg == True, axis=1, keepdims=True)
    false_neg_count = np.count_nonzero(false_neg == True, axis=1, keepdims=True)
    #
    accuracy = (true_pos_count + true_neg_count)/Y.shape[1]
    precision = true_pos_count/(true_pos_count + false_pos_count)
    recall = true_pos_count/(true_pos_count + false_neg_count)
    f1_score = 2*precision*recall/(precision + recall)

    np.set_printoptions(threshold=sys.maxsize)
    print('True positive: {}'.format(true_pos_count))
    print('False positive: {}'.format(false_pos_count))
    if printpath:
        print(str(filepaths[false_pos[0]]))
    print('True negative: {}'.format(true_neg_count))
    print('False negative: {}'.format(false_neg_count))
    if printpath:
        print(str(filepaths[false_neg[0]]))
        
    print('Accuracy: {}'.format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1_score: {}'.format(f1_score))
    
def find_errors(s, p, s_index = 0, p_index = 0, epsilon=0):
    if p_index >= len(p):
        return [-1 for _ in range(s_index, len(s))]
    diff = []
    if abs(s[s_index] - p[p_index]) <= epsilon:
        diff.append(0)
    else:
        diff.append(1)
        #diff.append(s[s_index] - p[p_index])
    diff.extend(find_errors(s, p, s_index + 1, p_index + 1, epsilon))
    if len(s) - s_index > len(p) - p_index:
        missing_diff = [-1]
        missing_diff.extend(find_errors(s, p, s_index + 1, p_index, epsilon))
        if missing_diff.count(0) > diff.count(0):
            diff = missing_diff
    return diff
    
def rms_normalize(x, rms_ref):
    rms = np.sqrt(np.mean(x*x))
    return rms_ref*(x/rms)
