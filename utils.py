#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import copy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import optimize
import statsmodels.api as sm

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.

    quant : float, default = 0.5
        Quantile to return (default is median).

    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.

    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.

    Returns
    -------
        epsilon: float

    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult

#### Accuracy Metrics ####
def MAE(X_true, X_fake, mask, n_miss):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X_fake[mask_] - X_true[mask_]).sum() / n_miss
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X_fake[mask_] - X_true[mask_]).sum() / n_miss

# For imputation, per sample find the minimum MAE
def MAE_min(X_true, X_fake, mask, n_miss):
    """
    Minimum Mean Absolute Error (MAE_min) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    X_fake : torch.DoubleTensor or np.ndarray, shape (nimp, n, d)
        Data with imputed variables.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n)
        Missing value mask (missing if True)

    n_miss: n*d missing values (needed since mask contains one-hot variables, so the size of mask is not the same as n*d)

    Returns
    -------
        MAE : float

    """
    # should be an ndarray
    nimp = X_fake.shape[0]
    M = np.sum(mask, axis=1) > 0
    abs_diff = np.absolute(np.expand_dims(X_true[M, :], axis=0) - X_fake[:, M, :]).sum(axis=2) # (nimp, n)
    min_abs_diff = np.min(abs_diff, axis=0) # (n)
    return min_abs_diff.sum() / n_miss # average over the n,d in the mask


def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())


# Functions to deal with categorical variables through one-hot encoding and decoding

# https://stackoverflow.com/questions/50607740/reverse-a-get-dummies-encoding-in-pandas
def undummify_(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col + prefix_sep)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def undummify(X, df_names_before, df_names_after):
    df = pd.DataFrame(X, columns = df_names_after) # to Pandas
    df = undummify_(df)[df_names_before]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.to_numpy()
    return df

def dummify(X, cat_indexes, divide_by=0, drop_first=False):
    df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
    df_names_before = df.columns
    for i in cat_indexes:
        df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=drop_first)
        if divide_by > 0: # needed for L1 distance to equal 1 when categories are different
            filter_col = [col for col in df if col.startswith(str(i) + '_')]
            df[filter_col] = df[filter_col] / divide_by
    df_names_after = df.columns
    df = df.to_numpy()
    return df, df_names_before, df_names_after

# Clip to prevent going beyond min-max and also round of the integers
def clip_extremes(X_real, X, int_indexes=None, min_max_clip=True):
    if int_indexes is not None:
        for i in int_indexes:
            X[:,i] = np.round(X[:,i], decimals=0)
    if min_max_clip:
        X_min = np.nanmin(X_real, axis=0, keepdims=1)
        X_max = np.nanmax(X_real, axis=0, keepdims=1)
        small = (X < X_min).astype(float)
        X = small*X_min + (1-small)*X
        big = (X > X_max).astype(float)
        X = big*X_max + (1-big)*X
    return X

# Function to generate samples until all possible classes are found (needed to prevent bugs in the metrics with crappy methods)
def try_until_all_classes_found(y, synthesizer, cat, max_tries=5):
    if not cat:
        for tries in range(max_tries):
            samples = synthesizer()
            samples_constant = sm.tools.tools.add_constant(samples)
            if samples_constant.shape != samples.shape: # this means it added the constant and thus, there is no constant already in the data
                return samples
    else:
        y_uniques = np.unique(y)
        for tries in range(max_tries):
            samples = synthesizer()
            y_fake_uniques = np.unique(samples[:, -1])
            samples_constant = sm.tools.tools.add_constant(samples)
            if np.isin(y_uniques, y_fake_uniques).all() and samples_constant.shape != samples.shape: # this means it added the constant and thus, there is no constant already in the data
                return samples
    return samples


# Mixed data is tricky, nearest neighboors (for the coverage) and Wasserstein distance (based on L2) are not scale invariant
# To ensure that the scaling between variables is relatively uniformized, we take inspiration from the Gower distance used in mixed-data KNNs: https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b
# Continuous: we do min-max normalization (to use Gower |x1-x2|/(max-min) as distance)
# Categorical: We one-hot and then divide by 2 (e.g., 0 0 0.5 with 0.5 0 0 will have distance 0.5 + 0.5 = 1)
# After these transformations, taking the L1 (City-block / Manhattan distance) norm distance will give the Gower distance
def minmax_scale_dummy(X_train, X_test, cat_indexes=[], mask=None, divide_by=2):
    X_train_ = copy.deepcopy(X_train)
    X_test_ = copy.deepcopy(X_test)
    # normalization of continuous variables
    scaler = MinMaxScaler()
    if len(cat_indexes) != X_train_.shape[1]: # if all variables are categorical, we do not scale-transform
        not_cat_indexes = [i for i in range(X_train_.shape[1]) if i not in cat_indexes]
        scaler.fit(X_train_[:, not_cat_indexes])

        #Transforms
        X_train_[:, not_cat_indexes] = scaler.transform(X_train_[:, not_cat_indexes])
        X_test_[:, not_cat_indexes] = scaler.transform(X_test_[:, not_cat_indexes])

    # One-hot the categorical variables (>=3 categories)
    df_names_before, df_names_after = None, None
    n = X_train.shape[0]
    if len(cat_indexes) > 0:
        X_train_test, df_names_before, df_names_after = dummify(np.concatenate((X_train_, X_test_), axis=0), cat_indexes, divide_by=divide_by)
        X_train_ = X_train_test[0:n,:]
        X_test_ = X_train_test[n:,:]

    # 1 2 3 4 6 4_1 4_2 7_1 7_2 7_3

    # We get the new mask now that there are one-hot features
    if mask is not None:
        if len(cat_indexes) == 0:
            return X_train_, X_test_, mask, scaler, df_names_before, df_names_after
        else:
            mask_new = np.zeros(X_train_.shape)
            for i, var_name in enumerate(df_names_after):
                if '_' in var_name: # one-hot
                    var_ind = int(var_name.split('_')[0])
                else:
                    var_ind = int(var_name)
                mask_new[:, i] = mask[:, var_ind]
            return X_train_, X_test_, mask_new, scaler, df_names_before, df_names_after
    else:
        return X_train_, X_test_, scaler, df_names_before, df_names_after

def minmax_scale(X_train, X_test, cat_indexes=[]):
    X_train_ = copy.deepcopy(X_train)
    X_test_ = copy.deepcopy(X_test)
    # normalization of continuous variables
    scaler = MinMaxScaler()
    if len(cat_indexes) != X_train_.shape[1]: # if all variables are categorical, we do not scale-transform
        not_cat_indexes = [i for i in range(X_train_.shape[1]) if i not in cat_indexes]
        scaler.fit(X_train_[:, not_cat_indexes])

        #Transforms
        X_train_[:, not_cat_indexes] = scaler.transform(X_train_[:, not_cat_indexes])
        X_test_[:, not_cat_indexes] = scaler.transform(X_test_[:, not_cat_indexes])

    return X_train_, X_test_, scaler
