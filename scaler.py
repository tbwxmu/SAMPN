from typing import Any

import numpy as np

class minmaxScaler:
    """

    https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    """
    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None, mins: np.ndarray=None,maxs: np.ndarray=None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token
        self.min=mins
        self.max=maxs

    def fit(self, X) -> 'StandardScaler':
        """ a=np.array([1,2,3,None],dtype=np.float32)
            b=np.nanmean(a)
            print(f'a:{a}\tb:{b}')
        """
        if type(X) is list:
            X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.mins=np.nanmin(X, axis=0)
        self.maxs=np.nanmax(X, axis=0)
        self.means = np.where(np.isnan(self.means), None, self.means)
        return self

    def transform(self, X):
        if type(X) is list:
            X = np.array(X).astype(float)
        transformed_with_nan = (X - self.mins) / (self.maxs - self.mins)
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        if type(X) is list:
            X = np.array(X).astype(float)
        transformed_with_nan = X * (self.maxs - self.mins) + self.mins
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
        return transformed_with_none

class StandardScaler:
    """A StandardScaler normalizes a dataset.

    When fit on a dataset, the StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the StandardScaler subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        Initialize StandardScaler, optionally with means and standard deviations precomputed.

        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: The token to use in place of nans.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis.

        :param X: A list of lists of floats.
        :return: The fitted StandardScaler.
        """
        if type(X) is list:
            X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats.
        :return: The transformed data.
        """
        if type(X) is list:
            X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data.
        """
        if type(X) is list:
            X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
