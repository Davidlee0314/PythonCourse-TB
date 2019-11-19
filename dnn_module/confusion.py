import os

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, precision_recall_curve
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          file_name='file_name'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # plt.show()
    os.makedirs('./confusion_matrix/', exist_ok=True)
    plt.savefig('./confusion_matrix/{}.png'.format(file_name))
    plt.cla()
    plt.clf()
    plt.close()

def show_data(cm, print_res = 0):
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    total = tp + fn + fp + tn
    if print_res == 1:
        print('\tPrecision     = {:.3f} ({} "Pred 1  " / {} "Actual 1")'.format(tp/(tp+fp), tp, tp+fp))
        print('\tRecall (TPR)  = {:.3f} ({} "Actual 1" / {} "Pred   1")'.format(tp/(tp+fn), tp, tp+fn))
        print('\tFallout (FPR) = {:.3f} ({} / {})'.format(fp/(fp+tn), fp, fp+tn))
        print('')
        print('\tTrue Positive  = {}'.format(tp))
        print('\tTrue Negative  = {}'.format(tn))
        print('\tFalse Positive = {}'.format(fp))
        print('\tFalse Negative = {}'.format(fn))
        print('')
        print('\tPredict Negative (0) = {:.3f} ({} / {})'.format( (tn+fn)/total, tn+fn, total ))
        print('\tPredict Positive (1) = {:.3f} ({} / {})'.format( (tp+fp)/total, tp+fp, total ))
        print('\tAccuracy             = {:.3f} ({} "Pred correct"/ {} "Total")'.format( (tp+tn)/total, tp+tn, total ))
    return tp/(tp+fp), tp/(tp+fn), fp/(fp+tn)

def cm_f1_score(labels, preds, file_name='file_name'):
    cm = confusion_matrix(labels, preds)  # labels df
    plot_confusion_matrix(cm, ['0', '1'], file_name=file_name)
    pr, tpr, fpr = show_data(cm, print_res = 1)
    f1 = 2 * (pr * tpr) / (pr + tpr)
    return f1

if __name__ == '__main__':
    y_pred = save_y_pred.copy() # df csv output
    thresh = 0.5
    y_pred [y_pred > thresh] = 1
    y_pred [y_pred <= thresh] = 0
    cm = confusion_matrix(y_train, y_pred)  # y_train df
    plot_confusion_matrix(cm, ['0', '1'], )
    pr, tpr, fpr = show_data(cm, print_res = 1)
    print('f1_score: ' + str(2 * (pr * tpr) / (pr + tpr)))