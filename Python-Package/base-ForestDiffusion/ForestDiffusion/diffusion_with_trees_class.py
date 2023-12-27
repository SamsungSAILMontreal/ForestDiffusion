import math
import numpy as np
from ForestDiffusion.utils.diffusion import VPSDE, get_pc_sampler
import copy
import xgboost as xgb
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from ForestDiffusion.utils.utils_diffusion import build_data_xt, euler_solve, IterForDMatrix, get_xt
from joblib import delayed, Parallel
from scipy.special import softmax

## Class for the flow-matching or diffusion model
# Categorical features should be numerical (rather than strings), make sure to use x = pd.factorize(x)[0] to make them as such
# Make sure to specific which features are categorical and which are integers
# Note: Binary features can be considered integers since they will be rounded to the nearest integer and then clipped
class ForestDiffusionModel():
  def __init__(self, 
               X, # Numpy dataset 
               label_y=None, # must be a categorical/binary variable; if provided will learn multiple models for each label y
               n_t=50, # number of noise level
               model='xgboost', # xgboost, random_forest, lgbm, catboost
               diffusion_type='flow', # vp, flow (flow is better, but only vp can be used for imputation)
               max_depth = 7, n_estimators = 100, eta=0.3, # xgboost hyperparameters
               tree_method='hist', reg_alpha=0.0, reg_lambda = 0.0, subsample=1.0, # xgboost hyperparameters
               num_leaves=31, # lgbm hyperparameters
               duplicate_K=100, # number of different noise sample per real data sample
               bin_indexes=[], # vector which indicates which column is binary
               cat_indexes=[], # vector which indicates which column is categorical (>=3 categories)
               int_indexes=[], # vector which indicates which column is an integer (ordinal variables such as number of cats in a box)
               remove_miss=False, # If True, we remove the missing values, this allow us to train the XGBoost using one model for all predictors; otherwise we cannot do it
               p_in_one=True, # When possible (when there are no missing values), will train the XGBoost using one model for all predictors
               true_min_max_values=None, # Vector of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
               gpu_hist=False, # using GPU or not with xgboost
               n_z=10, # number of noise to use in zero-shot classification
               eps=1e-3, 
               beta_min=0.1, 
               beta_max=8, 
               n_jobs=-1, # cpus used (feel free to limit it to something small, this will leave more cpus per model; for lgbm you have to use n_jobs=1, otherwise it will never finish)
               n_batch=1, # If >0 use the data iterator with the specified number of batches
               seed=666,
               **xgboost_kwargs): # you can pass extra parameter for xgboost

    assert isinstance(X, np.ndarray), "Input dataset must be a Numpy array"
    assert len(X.shape)==2, "Input dataset must have two dimensions [n,p]"
    assert diffusion_type == 'vp' or diffusion_type == 'flow'
    np.random.seed(seed)

    # Sanity check, must remove observations with only missing data
    obs_to_remove = np.isnan(X).all(axis=1)
    X = X[~obs_to_remove]
    if label_y is not None:
      label_y = label_y[~obs_to_remove]

    # Remove all missing values
    obs_to_remove = np.isnan(X).any(axis=1)
    if remove_miss or (obs_to_remove.sum() == 0):
      X = X[~obs_to_remove]
      if label_y is not None:
        label_y = label_y[~obs_to_remove]
      self.p_in_one = p_in_one # All variables p can be predicted simultaneously
    else:
      self.p_in_one = False

    int_indexes = int_indexes + bin_indexes # since we round those, we do not need to dummy-code the binary variables

    if true_min_max_values is not None:
        self.X_min = true_min_max_values[0]
        self.X_max = true_min_max_values[1]
    else:
        self.X_min = np.nanmin(X, axis=0, keepdims=1)
        self.X_max = np.nanmax(X, axis=0, keepdims=1)

    self.cat_indexes = cat_indexes
    self.int_indexes = int_indexes
    if len(self.cat_indexes) > 0:
        X, self.X_names_before, self.X_names_after = self.dummify(X) # dummy-coding for categorical variables

    # min-max normalization, this applies to dummy-coding too to ensure that they become -1 or +1
    self.scaler = MinMaxScaler(feature_range=(-1, 1))
    X = self.scaler.fit_transform(X)

    X1 = X
    self.X1 = copy.deepcopy(X1)
    self.b, self.c = X1.shape
    self.n_t = n_t
    self.duplicate_K = duplicate_K
    self.model = model
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.seed = seed
    self.num_leaves = num_leaves
    self.eta = eta
    self.gpu_hist = gpu_hist
    self.label_y = label_y
    self.n_jobs = n_jobs
    self.tree_method = tree_method
    self.reg_lambda = reg_lambda
    self.reg_alpha = reg_alpha
    self.subsample = subsample
    self.n_z = n_z
    self.xgboost_kwargs = xgboost_kwargs

    if model == 'random_forest' and np.sum(np.isnan(X1)) > 0:
      raise Error('The dataset must not contain missing data in order to use model=random_forest')

    self.diffusion_type = diffusion_type
    self.sde = None
    self.eps = eps
    self.beta_min = beta_min
    self.beta_max = beta_max
    if diffusion_type == 'vp':
      self.sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

    self.n_batch = n_batch
    if self.n_batch == 0: 
      if duplicate_K > 1: # we duplicate the data multiple times, so that X0 is k times bigger so we have more room to learn
        X1 = np.tile(X1, (duplicate_K, 1))

      X0 = np.random.normal(size=X1.shape) # Noise data

      # Make Datasets of interpolation
      X_train, y_train = build_data_xt(X0, X1, n_t=self.n_t, diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)

    if self.label_y is not None:
      assert np.sum(np.isnan(self.label_y)) == 0 # cannot have missing values in the label (just make a special categorical for nan if you need)
      self.y_uniques, self.y_probs = np.unique(self.label_y, return_counts=True)
      self.y_probs = self.y_probs/np.sum(self.y_probs)
      self.mask_y = {} # mask for which observations has a specific value of y
      for i in range(len(self.y_uniques)):
        self.mask_y[self.y_uniques[i]] = np.zeros(self.b, dtype=bool)
        self.mask_y[self.y_uniques[i]][self.label_y == self.y_uniques[i]] = True
        if self.n_batch == 0: 
          self.mask_y[self.y_uniques[i]] = np.tile(self.mask_y[self.y_uniques[i]], (duplicate_K))
    else: # assuming a single unique label 0
      self.y_probs = np.array([1.0])
      self.y_uniques = np.array([0])
      self.mask_y = {} # mask for which observations has a specific value of y
      self.mask_y[0] = np.ones(X1.shape[0], dtype=bool)

    if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
      rows_per_batch = self.b // self.n_batch
      batches = [rows_per_batch for i in range(self.n_batch-1)] + [self.b - rows_per_batch*(self.n_batch-1)]
      X1_splitted = {}
      for i in self.y_uniques:
        X1_splitted[i] = np.split(X1[self.mask_y[i], :], batches, axis=0)

    # Fit model(s)
    n_steps = n_t
    n_y = len(self.y_uniques) # for each class train a seperate model
    t_levels = np.linspace(eps, 1, num=n_t)

    if self.p_in_one:
      if self.n_jobs == 1:
        self.regr = [[None for i in range(n_steps)] for j in self.y_uniques]
        for i in range(n_steps):
          for j in range(len(self.y_uniques)):
              if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
                self.regr[j][i] = self.train_iterator(X1_splitted[j], t=t_levels[i], dim=None, i=i, j=j, k=self.c)
              else:
                self.regr[j][i] = self.train_parallel(
                X_train.reshape(self.n_t, self.b*self.duplicate_K, self.c)[i][self.mask_y[j], :], 
                y_train.reshape(self.b*self.duplicate_K, self.c)[self.mask_y[j], :]
                )
      else:
        if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
          self.regr = Parallel(n_jobs=self.n_jobs)(delayed(self.train_iterator)(X1_splitted[j], t=t_levels[i], dim=None, i=i, j=j, k=self.c) for i in range(n_steps) for j in self.y_uniques)
        else:
          self.regr = Parallel(n_jobs=self.n_jobs)( # using all cpus
                  delayed(self.train_parallel)(
                    X_train.reshape(self.n_t, self.b*self.duplicate_K, self.c)[i][self.mask_y[j], :], 
                    y_train.reshape(self.b*self.duplicate_K, self.c)[self.mask_y[j], :]
                    ) for i in range(n_steps) for j in self.y_uniques
                  )
        # Replace fits with doubly loops to make things easier
        self.regr_ = [[None for i in range(n_steps)] for j in self.y_uniques]
        current_i = 0
        for i in range(n_steps):
          for j in range(len(self.y_uniques)): 
            self.regr_[j][i] = self.regr[current_i]
            current_i += 1
        self.regr = self.regr_
    else:
      if self.n_jobs == 1:
        self.regr = [[[None for k in range(self.c)] for i in range(n_steps)] for j in self.y_uniques]
        for i in range(n_steps):
          for j in range(len(self.y_uniques)): 
            for k in range(self.c): 
              if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
                self.regr[j][i][k] = self.train_iterator(X1_splitted[j], t=t_levels[i], dim=k, i=i, j=j, k=k)
              else:
                self.regr[j][i][k] = self.train_parallel(
                X_train.reshape(self.n_t, self.b*self.duplicate_K, self.c)[i][self.mask_y[j], :], 
                y_train.reshape(self.b*self.duplicate_K, self.c)[self.mask_y[j], k]
                )
      else:
        if self.n_batch > 0: # Data iterator, no need to duplicate, not make xt yet
          self.regr = Parallel(n_jobs=self.n_jobs)(delayed(self.train_iterator)(X1_splitted[j], t=t_levels[i], dim=k, i=i, j=j, k=k) for i in range(n_steps) for j in self.y_uniques for k in range(self.c))
        else:
          self.regr = Parallel(n_jobs=self.n_jobs)( # using all cpus
                  delayed(self.train_parallel)(
                    X_train.reshape(self.n_t, self.b*self.duplicate_K, self.c)[i][self.mask_y[j], :], 
                    y_train.reshape(self.b*self.duplicate_K, self.c)[self.mask_y[j], k]
                    ) for i in range(n_steps) for j in self.y_uniques for k in range(self.c)
                  )
        # Replace fits with doubly loops to make things easier
        self.regr_ = [[[None for k in range(self.c)] for i in range(n_steps)] for j in self.y_uniques]
        current_i = 0
        for i in range(n_steps):
          for j in range(len(self.y_uniques)): 
            for k in range(self.c): 
              self.regr_[j][i][k] = self.regr[current_i]
              current_i += 1
        self.regr = self.regr_

  def train_iterator(self, X1_splitted, t, dim, i=0, j=0, k=0):
    np.random.seed(self.seed)

    it = IterForDMatrix(X1_splitted, t=t, dim=dim, n_batch=self.n_batch, n_epochs=self.duplicate_K, diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)
    data_iterator = xgb.QuantileDMatrix(it)

    xgb_dict = {'objective': 'reg:squarederror', 'eta': self.eta, 'max_depth': self.max_depth,
          "reg_lambda": self.reg_lambda, 'reg_alpha': self.reg_alpha, "subsample": self.subsample, "seed": self.seed, 
          "tree_method": self.tree_method, 'device': 'cuda' if self.gpu_hist else 'cpu', 
          "device": "cuda" if self.gpu_hist else 'cpu'}
    for myarg in self.xgboost_kwargs:
      xgb_dict[myarg] = self.xgboost_kwargs[myarg]
    out = xgb.train(xgb_dict, data_iterator, num_boost_round=self.n_estimators)

    return out

  def train_parallel(self, X_train, y_train):

    if self.model == 'random_forest':
      out = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.seed)
    elif self.model == 'lgbm':
      out = LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves, learning_rate=0.1, random_state=self.seed, force_col_wise=True)
    elif self.model == 'catboost':
      out = CatBoostRegressor(iterations=self.n_estimators, loss_function='RMSE', max_depth=self.max_depth, silent=True,
        l2_leaf_reg=0.0, random_seed=self.seed) # consider t as a golden feature if t is a variable
    elif self.model == 'xgboost':
      out = xgb.XGBRegressor(n_estimators=self.n_estimators, objective='reg:squarederror', eta=self.eta, max_depth=self.max_depth, 
        reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, subsample=self.subsample, seed=self.seed, tree_method=self.tree_method, 
        device='cuda' if self.gpu_hist else 'cpu', **self.xgboost_kwargs)
    else:
      raise Exception("model value does not exists")

    if len(y_train.shape) == 1:
      y_no_miss = ~np.isnan(y_train)
      out.fit(X_train[y_no_miss, :], y_train[y_no_miss])
    else:
      out.fit(X_train, y_train)

    return out

  def dummify(self, X):
    df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
    df_names_before = df.columns
    for i in self.cat_indexes:
      df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=True)
    df_names_after = df.columns
    df = df.to_numpy()
    return df, df_names_before, df_names_after

  def unscale(self, X):
    if self.scaler is not None: # unscale the min-max normalization
      X = self.scaler.inverse_transform(X)
    return X

  # Rounding for the categorical variables which are dummy-coded and then remove dummy-coding
  def clean_onehot_data(self, X):
    if len(self.cat_indexes) > 0: # ex: [5, 3] and X_names_after [gender_a gender_b cartype_a cartype_b cartype_c]
      X_names_after = copy.deepcopy(self.X_names_after.to_numpy())
      prefixes = [x.split('_')[0] for x in self.X_names_after if '_' in x] # for all categorical variables, we have prefix ex: ['gender', 'gender']
      unique_prefixes = np.unique(prefixes) # uniques prefixes
      for i in range(len(unique_prefixes)):
        cat_vars_indexes = [unique_prefixes[i] + '_' in my_name for my_name in self.X_names_after]
        cat_vars_indexes = np.where(cat_vars_indexes)[0] # actual indexes
        cat_vars = X[:, cat_vars_indexes] # [b, c_cat]
        # dummy variable, so third category is true if all dummies are 0
        cat_vars = np.concatenate((np.ones((cat_vars.shape[0], 1))*0.5,cat_vars), axis=1)
        # argmax of -1, -1, 0 is 0; so as long as they are below 0 we choose the implicit-final class
        max_index = np.argmax(cat_vars, axis=1) # argmax across all the one-hot features (most likely category)
        X[:, cat_vars_indexes[0]] = max_index
        X_names_after[cat_vars_indexes[0]] = unique_prefixes[i] # gender_a -> gender
      df = pd.DataFrame(X, columns = X_names_after) # to Pandas
      df = df[self.X_names_before] # remove all gender_b, gender_c and put everything in the right order
      X = df.to_numpy()
    return X

  # Unscale and clip to prevent going beyond min-max and also round of the integers
  def clip_extremes(self, X):
    if self.int_indexes is not None:
      for i in self.int_indexes:
        X[:,i] = np.round(X[:,i], decimals=0)
    small = (X < self.X_min).astype(float)
    X = small*self.X_min + (1-small)*X
    big = (X > self.X_max).astype(float)
    X = big*self.X_max + (1-big)*X
    return X

  def predict_over_c(self, X, i, j, k, dmat, expand=False):
    if dmat:
      X_used = xgb.DMatrix(data=X)
    else:
      X_used = X
    if k is None:
      return self.regr[j][i].predict(X_used)
    elif expand:
        return np.expand_dims(self.regr[j][i][k].predict(X_used), axis=1) # [b, 1]
    else:
      return self.regr[j][i][k].predict(X_used)

  # Return the score-fn or ode-flow output
  def my_model(self, t, y, mask_y=None, dmat=False, unflatten=True):
    if unflatten:
      # y is [b*c]
      c = self.c
      b = y.shape[0] // c
      X = y.reshape(b, c) # [b, c]
    else:
      X = y

    # Output
    out = np.zeros(X.shape) # [b, c]
    i = int(round(t*(self.n_t-1)))
    for j, label in enumerate(self.y_uniques):
      if mask_y[label].sum() > 0:
        if self.p_in_one:
          out[mask_y[label], :] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=None, dmat=dmat)
        else:
          for k in range(self.c):
            out[mask_y[label], k] = self.predict_over_c(X=X[mask_y[label], :], i=i, j=j, k=k, dmat=dmat)

    if self.diffusion_type == 'vp':
      alpha_, sigma_ = self.sde.marginal_prob_coef(X, t)
      out = - out / sigma_
    if unflatten:
      out = out.reshape(-1) # [b*c]
    return out


  # Generate new data by solving the reverse ODE/SDE
  def generate(self, batch_size=None, n_t=None):

    # Generate prior noise
    y0 = np.random.normal(size=(self.b if batch_size is None else batch_size, self.c))

    # Generate random labels
    label_y = self.y_uniques[np.argmax(np.random.multinomial(1, self.y_probs, size=y0.shape[0]), axis=1)]
    mask_y = {} # mask for which observations has a specific value of y
    for i in range(len(self.y_uniques)):
      mask_y[self.y_uniques[i]] = np.zeros(y0.shape[0], dtype=bool)
      mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True
    my_model = partial(self.my_model, mask_y=mask_y, dmat=self.n_batch > 0)

    if self.diffusion_type == 'vp':
      sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.n_t if n_t is None else n_t)
      ode_solved = get_pc_sampler(my_model, sde=sde, denoise=True, eps=self.eps)(y0.reshape(-1))
    else:
      ode_solved = euler_solve(my_model=my_model, y0=y0.reshape(-1), N=self.n_t if n_t is None else n_t) # [t, b*c]
    solution = ode_solved.reshape(y0.shape[0], self.c) # [b, c]
    solution = self.unscale(solution)
    solution = self.clean_onehot_data(solution)
    solution = self.clip_extremes(solution)

    # Concatenate y label if needed
    if self.label_y is not None:
      solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1) 

    return solution

  # Impute missing data by solving the reverse ODE while keeping the non-missing data intact
  def impute(self, k=1, X=None, label_y=None, repaint=False, r=5, j=0.1, n_t=None): # X is data with missing values
    assert self.diffusion_type != 'flow' # cannot use with flow=matching
  
    if X is None:
      X = self.X1
    if label_y is None:
      label_y = self.label_y
    if n_t is None:
      n_t = self.n_t

    if self.diffusion_type == 'vp':
      sde = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=n_t)

    if label_y is None: # single category 0
      mask_y = {}
      mask_y[0] = np.ones(X.shape[0], dtype=bool)
    else:
      mask_y = {} # mask for which observations has a specific value of y
      for i in range(len(self.y_uniques)):
        mask_y[self.y_uniques[i]] = np.zeros(X.shape[0], dtype=bool)
        mask_y[self.y_uniques[i]][label_y == self.y_uniques[i]] = True

    my_model = partial(self.my_model, mask_y=mask_y, dmat=self.n_batch > 0)

    for i in range(k):
      y0 = np.random.normal(size=X.shape)
      if self.diffusion_type == 'vp':
        ode_solved = get_pc_sampler(my_model, sde=sde, denoise=True, repaint=repaint, eps=self.eps, X_miss=X.reshape(-1))(y0.reshape(-1), r=r, j=int(math.ceil(j*n_t)))
      solution = ode_solved.reshape(y0.shape[0], self.c) # [b, c]
      solution = self.unscale(solution)
      solution = self.clean_onehot_data(solution)
      solution = self.clip_extremes(solution)
      # Concatenate y label if needed
      if self.label_y is not None:
        solution = np.concatenate((solution, np.expand_dims(label_y, axis=1)), axis=1) 
      if i == 0:
        imputed_data = np.expand_dims(solution, axis=0)
      else:
        imputed_data = np.concatenate((imputed_data, np.expand_dims(solution, axis=0)), axis=0)
    return imputed_data[0] if k==1 else imputed_data

  # Zero-shot classification of one batch
  def zero_shot_classification(self, x, n_t=10, n_z=10):
    assert self.label_y is not None # must have label conditioning to work

    h = 1 / n_t
    num_classes = len(self.y_uniques)
    L2_dist = []
    for i in range(num_classes): # for each class

      # Class conditioning
      mask_y = {}
      for k in range(len(self.y_uniques)):
        if k == i:
          mask_y[self.y_uniques[k]] = np.ones(x.shape[0], dtype=bool)
        else:
          mask_y[self.y_uniques[k]] = np.zeros(x.shape[0], dtype=bool)

      L2_dist_ = []
      for k in range(n_z): # monte-carlo over multiple noises
        t = 0
        for j in range(n_t-1): # averaging over multiple noise levels [t=1/n, ... (n-1)/n]
          t = t + h
          np.random.seed(10000*k + j)
          y0 = np.random.normal(size=x.shape)
          xt = get_xt(x1=x, t=t, x0=y0, dim=None, diffusion_type=self.diffusion_type, eps=self.eps, sde=self.sde)[0]
          pred_ = self.my_model(t=t, y=xt, mask_y=mask_y, unflatten=False, dmat=self.n_batch > 0)
          if self.diffusion_type == 'flow':
            x0 = x - pred_ # x0 = x1 - (x1 - x0)
          elif self.diffusion_type == 'vp':
            x0 = pred_ # x0
          L2_dist_ += [np.expand_dims(np.sum((x0 - y0) ** 2, axis=1), axis=0)] # [1, b]
      L2_dist += [np.concatenate(L2_dist_, axis=0)] # [n_z*n_t, b]

    # Based on absolute
    L2_abs = []
    for i in range(num_classes): # for each class
      L2_abs += [np.expand_dims(np.mean(L2_dist[i], axis=0), axis=0)] # [1, b]
    L2_abs = np.concatenate(L2_abs, axis=0) # [c, b]
    prob_avg = softmax(-L2_abs, axis=0) # [b]
    most_likely_class_avg = np.argmin(L2_abs, axis=0) # [b]
    return self.y_uniques[most_likely_class_avg], prob_avg

  # Zero-shot classification using https://diffusion-classifier.github.io/static/docs/DiffusionClassifier.pdf
  # Return the absolute and relative accuracies
  def predict(self, X, n_t=None, n_z=None):
    if n_t is None:
      n_t = self.n_t
    if n_z is None:
      n_z = self.n_z

    # Data transformation (assuming we get the raw data)
    if len(self.cat_indexes) > 0:
      X, _, _ = self.dummify(X) # dummy-coding for categorical variables
    X = self.scaler.transform(X)

    most_likely_class_avg, prob_avg = self.zero_shot_classification(X, n_t=n_t, n_z=n_z)

    return most_likely_class_avg

  def predict_proba(self, X, n_t=None, n_z=None):
    if n_t is None:
      n_t = self.n_t
    if n_z is None:
      n_z = self.n_z

    # Data transformation (assuming we get the raw data)
    if len(self.cat_indexes) > 0:
      X, _, _ = self.dummify(X) # dummy-coding for categorical variables
    X = self.scaler.transform(X)

    most_likely_class_avg, prob_avg = self.zero_shot_classification(X, n_t=n_t, n_z=n_z)

    return prob_avg