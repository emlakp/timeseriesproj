from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def read_data(path):
    data = pd.read_csv(path, parse_dates=['Datetime'], index_col=['Datetime'])
    return data


def split(data, split, scaler):
    '''

    :param data: time series
    :param split: percent of data contained in the training set
    :param scaler: scaling method to apply
    :return:
    '''

    scalermm = MinMaxScaler()
    scalerstd = StandardScaler()

    split = int(split * (len(data)))
    train = data[:split]
    test = data[split:]

    # note that we apply scaling fit on training data to test set
    if (scaler.lower() == 'minmax'):
        train = scalermm.fit_transform(train)
        test = scalermm.transform(test)
        return train, test, scalermm
    elif (scaler.lower() == 'standard'):
        train = scalerstd.fit_transform(train)
        test = scalerstd.transform(test)
        return train, test, scalerstd


def gen_windows(data, n_lags, n_pred_values, n_features=6):
    features, labels = [], []
    for i in range(len(data)):
        lag_end = i + n_lags
        pred_end = lag_end + n_pred_values
        if pred_end > len(data):
            break
        lags, targets = data[i:lag_end,:], data[lag_end:pred_end,:]
        features.append(lags)
        labels.append(targets)

    features = np.array(features)
    labels = np.array(labels)

    features = features.reshape((features.shape[0], features.shape[1], n_features))
    labels = labels.reshape((labels.shape[0], labels.shape[1], n_features))
    return features, labels
