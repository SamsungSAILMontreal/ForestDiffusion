import numpy as np
from ForestDiffusion.utils.diffusion import VPSDE
import xgboost as xgb

# Build the dataset of x(t) at multiple values of t 
def build_data_xt(x0, x1, x_covs=None, n_t=101, diffusion_type='flow', eps=1e-3, sde=None):
  b, c = x1.shape

  # Expand x0, x1
  x0 = np.expand_dims(x0, axis=0) # [1, b, c]
  x1 = np.expand_dims(x1, axis=0) # [1, b, c]

  # t and expand
  t = np.linspace(eps, 1, num=n_t)
  t_expand = np.expand_dims(t, axis=(1,2)) # [t, 1, 1]

  if diffusion_type == 'vp': # Forward diffusion from x0 to x1
    mean, std = sde.marginal_prob(x1, t_expand)
    x_t = mean + std*x0
  else: # Interpolation between x0 and x1
    x_t = t_expand * x1 + (1 - t_expand) * x0 # [t, b, c]
  x_t = x_t.reshape(-1,c) # [t*b, c]

  X = x_t

  # Output to predict
  if diffusion_type == 'vp':
    alpha_, sigma_ = sde.marginal_prob_coef(x1, t_expand)
    y = x0.reshape(b, c)
  else:
    y = x1.reshape(b, c) - x0.reshape(b, c) # [b, c]

  if x_covs is not None: # additional covariates
    c_new = x_covs.shape[1]
    X_covs = np.tile(np.expand_dims(x_covs, axis=0), (n_t, 1, 1)).reshape(-1,c_new)
    X = np.concatenate((X,X_covs), axis=1)

  return X, y

#### Below is for Flow-Matching Sampling ####

# Euler solver
def euler_solve(y0, my_model, N=101):
  h = 1 / (N-1)
  y = y0
  t = 0
  # from t=0 to t=1
  for i in range(N-1):
    y = y + h*my_model(t=t, y=y)
    t = t + h
  return y

# Get X[t], y where t is a scalar
def get_xt(x1, t, dim, diffusion_type='flow', eps=1e-3, sde=None, x0=None):
  b, c = x1.shape
  if x0 is None:
    x0 = np.random.normal(size=x1.shape) # Noise data

  if diffusion_type == 'vp': # Forward diffusion from x0 to x1
    mean, std = sde.marginal_prob(x1, t)
    x_t = mean + std*x0
  else: # Interpolation between x0 and x1
    x_t = t * x1 + (1 - t) * x0 # [b, c]

  # Output to predict
  if dim is None:
    if diffusion_type == 'vp':
      y = x0
    else:
      y = x1 - x0
  else:
    if diffusion_type == 'vp':
      y = x0[:, dim]
    else:
      y = x1[:, dim] - x0[:, dim]

  return x_t, y


# Seperate dataset into multiple batches for memory-efficient training
class IterForDMatrix(xgb.core.DataIter):
  """A data iterator for XGBoost DMatrix.

  `reset` and `next` are required for any data iterator, other functions here
  are utilites for demonstration's purpose.

  """

  def __init__(self, data, data_covs, t, dim, n_batch=1000, n_epochs=10, diffusion_type='flow', eps=1e-3, sde=None):
    self._data = data
    self._data_covs = data_covs
    self.n_batch = n_batch
    self.n_epochs = n_epochs
    self.t = t
    self.diffusion_type = diffusion_type
    self.eps = eps
    self.sde = sde
    self.dim = dim
    self.it = 0  # set iterator to 0
    super().__init__()

  def reset(self):
    """Reset the iterator"""
    self.it = 0

  def next(self, input_data):
    """Yield next batch of data."""
    if self.it == self.n_batch*self.n_epochs: # stops after k epochs
      return 0
    x_t, y = get_xt(x1=self._data[self.it % self.n_batch], dim=self.dim, t=self.t, diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)
    if self._data_covs is not None:
      x_t = np.concatenate((x_t, self._data_covs[self.it % self.n_batch]), axis=1)
    if len(y.shape) == 1:
      y_no_miss = ~np.isnan(y)
      input_data(data=x_t[y_no_miss, :], label=y[y_no_miss])
    else:
      input_data(data=x_t, label=y)
    self.it += 1
    return 1

#### Below is for Flow-Matching Sampling ####

# Euler solver
def euler_solve(y0, my_model, N=101):
  h = 1 / (N-1)
  y = y0
  t = 0
  # from t=0 to t=1
  for i in range(N-1):
    y = y + h*my_model(t=t, y=y)
    t = t + h
  return y
