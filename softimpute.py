#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.utils.extmath import randomized_svd

F32PREC = np.finfo(np.float32).eps

# convergence criterion for softimpute
def converged(x_old, x, mask, thresh):
    x_old_na = x_old[mask]
    x_na = x[mask]
    rmse = np.sqrt(np.sum((x_old_na - x_na) ** 2))
    denom = np.sqrt((x_old_na ** 2).sum())
    
    if denom == 0 or (denom < F32PREC and rmse > F32PREC):
      return False
    else:
      return (rmse / denom) < thresh


def softimpute(x, lamb, maxit = 1000, thresh = 1e-5):
    """
    x should have nan values (the mask is not provided as an argument)
    """
    mask = ~np.isnan(x)
    imp = x.copy()
    imp[~mask] = 0

    for i in range(maxit):
        if x.shape[0]*x.shape[1] > 1e6:
            U, d, V = randomized_svd(imp, n_components = np.minimum(200, x.shape[1]))
        else:
            U, d, V = np.linalg.svd(imp, compute_uv = True, full_matrices=False)
        d_thresh = np.maximum(d - lamb, 0)
        rank = (d_thresh > 0).sum()
        d_thresh = d_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        D_thresh = np.diag(d_thresh)
        res = np.dot(U_thresh, np.dot(D_thresh, V_thresh))
        if converged(imp, res, mask, thresh):
            break
        imp[~mask] = res[~mask]
        
    return U_thresh, imp


def test_x(x, mask):
    # generate additional missing values
    # such that each row has at least 1 observed value (assuming also x.shape[0] > x.shape[1])
    save_mask = mask.copy()
    for i in range(x.shape[0]):
      idx_obs = np.argwhere(save_mask[i, :] == 1).reshape((-1)) 
      if len(idx_obs) > 0:
          j = np.random.choice(idx_obs, 1)
          save_mask[i, j] = 0
    mmask = np.array(np.random.binomial(np.ones_like(save_mask), save_mask * 0.1), dtype=bool)
    xx = x.copy()
    xx[mmask] = np.nan
    return xx, mmask


def cv_softimpute(x, grid_len = 15, maxit = 1000, thresh = 1e-5):   
    # impute with constant
    mask = ~np.isnan(x)
    x0 = x.copy()
    #x0 = copy.deepcopy(x)
    x0[~mask] = 0
    # svd on x0
    if x.shape[0]*x.shape[1] > 1e6:
        _, d, _ = randomized_svd(x0, n_components = np.minimum(200, x.shape[1]))
    else:
        d = np.linalg.svd(x0, compute_uv=False, full_matrices=False)
    # generate grid for lambda values
    lambda_max = np.max(d)
    lambda_min = 0.001*lambda_max
    grid_lambda = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), grid_len).tolist())
  
    cv_error = []
    for lamb in grid_lambda:
        xx, mmask = test_x(x, mask)
        mmask = ~np.isnan(xx)
        _, res = softimpute(xx, lamb, maxit, thresh)
        cv_error.append(np.sqrt(np.nanmean((res.flatten() - x.flatten())**2)))

    return cv_error, grid_lambda

