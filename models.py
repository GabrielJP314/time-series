################################################################################
###                             Models functions                             ###
################################################################################

import numpy as np
from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression as LR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D as KerasConv1D, LSTM as KerasLSTM, Dense, Input

# Baseline models
class NaiveModel:
    """
        Naive model
    """
    def __init__(self):
        pass

    def fit(self, y_train):
        """
            Fit the model
        """
        self.y_train = y_train

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return np.repeat(self.y_train[-1], n_preds)
    
class SeasonalNaiveModel:
    """
        Seasonal Naive model
    """
    def __init__(self, period):
        self.period = period

    def fit(self, y_train):
        """
            Fit the model
        """
        self.y_train = y_train

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return np.tile(self.y_train[-self.period:], n_preds // self.period + 1)[:n_preds]
    
class AverageModel:
    """
        Average model
    """
    def __init__(self):
        pass

    def fit(self, y_train):
        """
            Fit the model
        """
        self.y_train = y_train

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return np.repeat(np.mean(self.y_train), n_preds)
    
class DriftModel:
    """
        Drift model
    """
    def __init__(self):
        pass

    def fit(self, y_train):
        """
            Fit the model
        """
        self.y_train = y_train

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return np.arange(1, n_preds + 1) * (self.y_train[-1] - self.y_train[0]) / len(self.y_train) + self.y_train[-1]
    
# Exponential smoothing models
class SimpleExpSmoothing:
    """
        Simple Exponential Smoothing model
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, y_train):
        """
            Fit the model
        """
        self.y_train = y_train
        self.y_pred = [y_train[0]]

        for i in range(1, len(y_train)):
            self.y_pred.append(self.alpha * y_train[i] + (1 - self.alpha) * self.y_pred[-1])

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return np.repeat(self.y_pred[-1], n_preds)
    
class Holt:
    """
        Holt model
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def fit(self, y_train):
        """
            Fit the model
        """
        self.y_train = y_train
        self.y_pred = [y_train[0]]
        self.b = y_train[1] - y_train[0]

        for i in range(1, len(y_train)):
            self.y_pred.append(self.alpha * y_train[i] + (1 - self.alpha) * (self.y_pred[-1] + self.b))
            self.b = self.beta * (self.y_train[i] - self.y_train[i - 1]) + (1 - self.beta) * self.b

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return np.repeat(self.y_pred[-1], n_preds)
    
class HoltWinters:
    """
        Holt-Winters model
    """
    def __init__(self, alpha, beta, gamma, period):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.period = period

    def fit(self, y_train):
        """
            Fit the model
        """
        self.y_train = y_train
        self.y_pred = [y_train[0]]
        self.b = y_train[1] - y_train[0]
        self.s = np.zeros(self.period)
        self.s[:self.period] = y_train[:self.period]

        for i in range(1, len(y_train)):
            self.y_pred.append(self.alpha * y_train[i] + (1 - self.alpha) * (self.y_pred[-1] + self.b))
            self.b = self.beta * (self.y_train[i] - self.y_train[i - 1]) + (1 - self.beta) * self.b
            self.s[i % self.period] = self.gamma * y_train[i] + (1 - self.gamma) * self.s[i % self.period]

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return np.repeat(self.y_pred[-1], n_preds)

# ARIMA models
class ARIMA:
    """
        ARIMA model
    """
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q
        self.fitted_values = None
        self.model = None

    def fit(self, y_train):
        """
            Fit the model
        """
        self.model = ARIMA_Model(y_train, order=(self.p, self.d, self.q)).fit()
        self.fitted_values = self.model.fittedvalues
        return self.model

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return self.model.forecast(steps=n_preds)
    
class SARIMA:
    """
        SARIMA model
    """
    def __init__(self, p, d, q, P, D, Q, s):
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s
        self.fitted_values = None
        self.model = None

    def fit(self, y_train):
        """
            Fit the model
        """
        self.model = SARIMAX(y_train, order=(self.p, self.d, self.q), seasonal_order=(self.P, self.D, self.Q, self.s)).fit()
        self.fitted_values = self.model.fittedvalues
        return self.model

    def predict(self, n_preds):
        """
            Predict n_preds
        """
        return self.model.forecast(steps=n_preds)
    
# Linear models
class LinearRegression:
    """
        Linear Regression model
    """
    def __init__(self):
        self.model = LR()

    def fit(self, X_train, y_train):
        """
            Fit the model
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
            Predict n_preds
        """
        return self.model.predict(X_test)
    
# Convolutional 1D models
class Conv1D:
    """
        Convolutional 1D model
    """
    def __init__(self, input_shape, filters, kernel_size):
        self.model = Sequential()
        self.model.add(KerasConv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        """
            Fit the model
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        """
            Predict n_preds
        """
        return self.model.predict(X_test)
    
# LSTM models
class LSTM:
    """
        LSTM model
    """
    def __init__(self, input_shape, units):
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))
        self.model.add(KerasLSTM(units=units))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X_train, y_train, epochs=10, batch_size=32, verbose=1):
        """
            Fit the model
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, X_test, verbose=1):
        """
            Predict one point to each sample in X_test
        """
        return self.model.predict(X_test, verbose=verbose)
    
    def predict_n_preds(self, X_test, n_preds):
        """
            Given a sample X_test, predict n_preds
        """
        preds = []
        for i in range(n_preds):
            pred = self.model.predict(X_test)
            preds.append(pred)
            X_test = np.concatenate([X_test[:, 1:], pred], axis=1)
        return np.array(preds).flatten()
