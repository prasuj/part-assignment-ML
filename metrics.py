from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    df3 =pd.DataFrame({
        "y":y,
        "y_hat":y_hat
        })
    df3 ['boolean'] = (df3['y']==df3['y_hat'])
    count_true,count_false = df3 ['boolean'].value_counts()
    total_samples = count_true + count_false
    accuracy = count_true / total_samples
    return accuracy
    
def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    assert y_hat.size == y.size
    df_confusion = pd.crosstab(y_hat, y)
    TP = df_confusion['Yes']['Yes']
    FP = df_confusion['No']['Yes']
    FN = df_confusion['Yes']['No']
    TN = df_confusion['No']['No']
    k = TP + FP
    return TP/k

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    assert y_hat.size == y.size
    df_confusion = pd.crosstab(y_hat, y)
    TP = df_confusion['Yes']['Yes']
    FP = df_confusion['No']['Yes']
    FN = df_confusion['Yes']['No']
    TN = df_confusion['No']['No']
    k = TP + FN
    return TP/k

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    arr_predicted = np.array(y_hat.values.tolist())
    arr_gt = np.array(y.values.tolist())
    arr_error = np.subtract(arr_predicted,arr_gt)
    arr_error = np.square(arr_error)
    tot_error = np.sum(arr_error)
    samples = arr_error.size
    mean_square = tot_error / samples
    root_mean_square_error = np.sqrt(mean_square)
    return root_mean_square_error


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    arr_predicted = np.array(y_hat.values.tolist())
    arr_gt = np.array(y.values.tolist())
    arr_error = np.subtract(arr_predicted,arr_gt)
    arr_error = np.absolute(arr_error)
    tot_error = arr_error.sum()
    samples = arr_error.size 
    mean_absolute_error = tot_error / samples
    return mean_absolute_error