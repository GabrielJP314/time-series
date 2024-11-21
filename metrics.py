################################################################################
###                            Metrics functions                             ###
################################################################################

import numpy as np

def mae(y_true, y_pred):
    """
        Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    """
        Mean Squared Error
    """
    return np.mean(np.square(y_true - y_pred))

def rmse(y_true, y_pred):
    """
        Root Mean Squared Error
    """
    return np.sqrt(mse(y_true, y_pred))

def mape(y_true, y_pred):
    """
        Mean Absolute Percentage Error
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    """
        Symmetric Mean Absolute Percentage Error
    """
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 200

def r2(y_true, y_pred):
    """
        R^2 score
    """
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (ss_res / ss_tot)

def mase(y_true, y_pred, y_train):
    """
        Mean Absolute Scaled Error
    """
    return mae(y_true, y_pred) / mae(y_train[1:], y_train[:-1])

def aic(y_true, y_pred, n_params):
    """
        Akaike Information Criterion
    """
    return len(y_true) * np.log(mse(y_true, y_pred)) + 2 * n_params

def bic(y_true, y_pred, n_params):
    """
        Bayesian Information Criterion
    """
    return len(y_true) * np.log(mse(y_true, y_pred)) + n_params * np.log(len(y_true))