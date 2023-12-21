
# functools.partial 
partial <- function(f, ...) {
  l <- list(...)
  function(...) {
    do.call(f, c(l, list(...)))
  }
}

# get mu and sigma from the SDE
marginal_prob_coef = function(t, beta_0=0.1, beta_1=8.0){
  log_mean_coeff = -0.25 * (t^2) * (beta_1 - beta_0) - (0.5 * t * beta_0)
  mu = exp(log_mean_coeff)
  std = sqrt(1 - exp(2. * log_mean_coeff))
  return(list(mu, std))
}

# Build the dataset of x(t) at multiple values of t 
build_data_xt = function(x0, x1, n_t=101, flow=FALSE, eps=1e-3, beta_0=0.1, beta_1=8.0){
  b = dim(x1)[[1]]
  c = dim(x1)[[2]]

  t = seq(eps, 1, length.out=n_t)

  x_t = vector("list", n_t)
  if (flow){ #Interpolation between x0 and x1
    for (i in 1:n_t){
      x_t[[i]] = t[i] * x1 + (1 - t[i]) * x0
    }
  } else{ # Forward diffusion from x0 to x1
    for (i in 1:n_t){
      mu_std = marginal_prob_coef(t[i], beta_0=beta_0, beta_1=beta_1)
      x_t[[i]] = mu_std[[1]]*x1 + mu_std[[2]]*x0
    }
  }

  # Output to predict
  if (flow){
    y = x1 - x0
  } else{
    y = x0
  }

  return(list(x_t, y))
}

#' @title Diffusion and Flow-based XGBoost Model for generating or imputing data
#' @description Train XGBoost regression models to estimate the score-function (for diffusion models) or the flow (flow-based models). These models can then be used to generate new fake samples or impute missing values.
#' @param X data.frame of the dataset to be used. 
#' @param n_cores number of cpu cores used (if NULL, it will use all cores, otherwise it will use min(n_cores, max_available_cores); using more cores makes training faster, but increases the memory cost (so reduce it if you have memory problems)
#' @param label_y optional vector containing the outcome variable if it is categorical for improved performance by training separate models per class; cannot contain missing values
#' @param name_y name of label_y
#' @param n_t number of noise levels (and sampling steps); increase for higher performance, but slows down training and sampling
#' @param flow If TRUE, uses flow (an ODE deterministic method); otherwise uses vp (a SDE stochastic method); 'vp' generally has slightly worse performance, but it is the only method that can be used for imputation
#' @param max_depth max depth of the trees per XGBoost model
#' @param n_estimators number of trees per XGBoost model
#' @param eta learning rate per XGBoost model
#' @param duplicate_K number of noise per sample (or equivalently the number of times the rows of the dataset are duplicated); should be as high as possible; higher values increase the memory demand
#' @param true_min_max_values (optional) list of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
#' @param eps minimum noise level
#' @param beta_min value of the beta_min in the vp process
#' @param beta_max value of the beta_max in the vp process
#' @param seed (optional) random seed used
#' @return Returns an object of the class "ForestDiffusion" which is list containing the XGBoost model fits
#' @examples
#'  \dontrun{
#'  data(iris)
#'  iris[,1:4] = missForest::prodNA(iris[,1:4], noNA = 0.2) # adding missing data
#'  X = data.frame(iris[,1:4])
#'  y = iris[,5]
#'  
#'  ## Generation
#'  
#'  # Classification problem (outcome is categorical)
#'  forest_model = ForestDiffusion(X=X, n_cores=1, label_y=y, n_t=50, duplicate_K=50, flow=TRUE)
#'  # last variable will be the label_y
#'  Xy_fake = ForestDiffusion.generate(forest_model, batch_size=NROW(iris))
#'  
#'  # When you do not want to train a seperate model per model (or you have a regression problem)
#'  Xy = X
#'  Xy$y = y
#'  forest_model = ForestDiffusion(X=Xy, n_cores=1, n_t=50, duplicate_K=50, flow=TRUE)
#'  Xy_fake = ForestDiffusion.generate(forest_model, batch_size=NROW(iris))
#'  
#'  ## Imputation
#'  
#'  # flow=TRUE generate better data but it cannot impute data
#'  forest_model = ForestDiffusion(X=Xy, n_cores=1, n_t=50, duplicate_K=50, flow=FALSE)
#'  nimp = 5 # number of imputations needed
#'  # regular (fast)
#'  Xy_fake = ForestDiffusion.impute(forest_model, k=nimp)
#'  # REPAINT (slow, but better)
#'  Xy_fake = ForestDiffusion.impute(forest_model, repaint=TRUE, r=10, j=5, k=nimp)
#'}
#'  
#' @import xgboost foreach parallel doParallel parallelly
#' @importFrom caret dummyVars
#' @importFrom stats predict
#' @references
#' Alexia Jolicoeur-Martineau, Kilian Fatras, Tal Kachman. 
#' Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees. 
#' arXiv:2309.09968.
#' @export
ForestDiffusion = function(X, 
               n_cores,
               label_y=NULL, # must be a categorical/binary variable; if provided will learn multiple models for each label y
               name_y = 'y',
               n_t=50,
               flow=TRUE, # if TRUE, use flow-matching; otherwise, use diffusion
               max_depth = 7, n_estimators = 100, eta=0.3, # xgboost hyperparameters
               duplicate_K=50, # number of noise per data sample (or equivalently the number of times we duplicate the rows of the dataset)
               true_min_max_values=NULL, # Vector of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping
               eps = 1e-3, 
               beta_min=0.1, 
               beta_max=8, 
               seed=NULL){

  if (!is.null(seed)) set.seed(seed)

  X = data.frame(X)

  # We must temporarely change the names to not have any '.' since we will need those to identify categorical variables
  names(X) = gsub('\\.', 'PlAcEhOlDeR', names(X))

  # Sanity check, must remove observations with only missing data
  obs_to_remove = rowSums(!is.na(X)) == 0 
  X = X[!obs_to_remove,]
  if (!is.null(label_y)) label_y = label_y[!obs_to_remove]

  b = NROW(X)
  c = NCOL(X)

  # vector that indicates which column is binary
  bin_indexes=c()
  cat_indexes=c()
  int_indexes=c()
  for (i in 1:NCOL(X)){
    if (is.factor(X[,i])){
      if (length(levels(X[,i]))==2) bin_indexes = c(bin_indexes, i) # vector that indicates which column is binary
      else cat_indexes = c(cat_indexes, i) # vector that indicates which column is categorical (>=3 categories)
    }
    else{
      # there is no decimals, these are integers
      if (sum(round(X[,i], 0) == X[,i], na.rm=TRUE) == NCOL(X)) int_indexes = c(int_indexes, i)
    }
  }

  int_indexes = c(int_indexes, bin_indexes) # since we round those

  # Processing to make all categorical variables as numeric and store the levels so we can easily revert them back to their original values
  cat_labels = vector("list", length(c(cat_indexes, bin_indexes)))
  cat_levels = vector("list", length(c(cat_indexes, bin_indexes)))
  j = 1
  for (i in sort(c(cat_indexes, bin_indexes))){ # from smallest index to largest index
    x_factor = factor(X[,i])
    
    unique_labels = unique(X[,i])
    cat_labels[[j]] = unique_labels[!is.na(unique_labels)]

    X[,i] =  as.numeric(x_factor)

    unique_levels = unique(X[,i])
    cat_levels[[j]] = unique_levels[!is.na(unique_levels)] # remove NA from uniques
    j = j + 1
  }
  # revert using factor(as.numeric(y), labels=levels(y))

  # min and max 
  if (is.null(true_min_max_values)){
    X_min = vector("numeric", NCOL(X))
    X_max = vector("numeric", NCOL(X))
    for (i in 1:NCOL(X)){
      X_min[i] = min(X[,i], na.rm=TRUE)
      X_max[i] = max(X[,i], na.rm=TRUE)
    }
  } else{
    X_min = true_min_max_values[[1]]
    X_max = true_min_max_values[[2]]
  }

  if (length(cat_indexes) > 0){ # dummy encoding for categorical variables
    for (i in cat_indexes){
      X[,i] =  as.factor(X[,i])
    }
    dummy <- caret::dummyVars(" ~ .", data=X, fullRank=TRUE)
    X <- data.frame(stats::predict(dummy, newdata = X))
  }

  # min and max 
  if (is.null(true_min_max_values)){
    X_min_ = vector("numeric", NCOL(X))
    X_max_ = vector("numeric", NCOL(X))
    for (i in 1:NCOL(X)){
      X_min_[i] = min(X[,i], na.rm=TRUE)
      X_max_[i] = max(X[,i], na.rm=TRUE)
    }
  } else{
    X_min_ = true_min_max_values[[1]]
    X_max_ = true_min_max_values[[2]]
  }

  # min-max normalization, this applies to dummies too to ensure that they become -1 or +1
  for (i in 1:NCOL(X)){
    X[,i] = ((X[,i] - X_min_[i])/(X_max_[i] - X_min_[i]))*2 - 1 # [-1 to 1]
  }

  X1 = X
  b = dim(X1)[1]
  c = dim(X1)[2]

  # we duplicate the data multiple times, so that X0 is k times bigger so we have more room to learn
  if (duplicate_K > 1) X1 = X1[rep(seq_len(NROW(X1)), each = duplicate_K), ]

  # Create noise data
  X0 = X1
  for (i in 1:NCOL(X)){
    X0[,i] = stats::rnorm(NROW(X1))
  }

  if (is.null(label_y)){
    y_probs = c(1.0)
    y_uniques = c(0)
    mask_y = vector("list", 1) # mask for which observations has a specific value of y
    mask_y[[1]] = rep(TRUE, NROW(X1))
  } else{
    if (sum(is.na(label_y)) > 0) stop("cannot have missing values in the label (just make a special categorical for nan if you need to)") 
    y_probs = table(label_y)/length(label_y)
    y_uniques = unique(label_y)
    mask_y = vector("list", length(y_uniques)) # mask for which observations has a specific value of y
    for (i in 1:length(y_uniques)){
      mask_y[[i]] = rep(FALSE, NROW(X))
      mask_y[[i]][label_y == y_uniques[i]] = TRUE
      mask_y[[i]] = mask_y[[i]][rep(seq_len(NROW(X)), each = duplicate_K)]
    }
    #label_y = label_y[rep(seq_len(NROW(X)), each = duplicate_K)]
  }

  # Make Datasets of interpolation
  X_train_y_train = build_data_xt(X0, X1, n_t=n_t, flow=flow, eps=eps, beta_0=beta_min, beta_1=beta_max)
  X_train = X_train_y_train[[1]]
  y_train = X_train_y_train[[2]]

  params <- list(booster = "gbtree", objective = "reg:squarederror", max_depth=max_depth, eta=eta, lambda=0, tree_method  = "hist")

  # Fit model(s)
  n_steps = n_t
  n_y = length(y_uniques) # for each class train a seperate model
  if (is.null(n_cores)){
    n_cores = parallelly::availableCores(omit = 1)
    } else{
      n_cores = min(n_cores, parallelly::availableCores(omit = 1))
    }
  if (n_cores == 1){
    regr = vector('list', n_steps)
    for(i in 1:n_steps){
      regr[[i]] = vector('list', n_y)
      for(j in 1:n_y){
        regr[[i]][[j]] = vector('list', c)
      } 
    }
    for(i in 1:n_steps){
        for(j in 1:n_y){
            for(k in 1:c){
                # Training function
                train_xgboost = function(X_train, y_train, params, n_estimators=100, seed=666){
                  # Remove observations with missing values in y
                  obs_to_remove = is.na(y_train)
                  X_train_ = X_train[!obs_to_remove,]
                  y_train_ = y_train[!obs_to_remove]
                  set.seed(seed)
                  out = xgboost::xgboost(as.matrix(X_train_), y_train_, params=params, nrounds=n_estimators, verbose=0)
                  return(out)
                }
                regr[[i]][[j]][[k]] = train_xgboost(X_train[[i]][mask_y[[j]], ], y_train[mask_y[[j]], k], params, n_estimators=n_estimators, seed=seed)
            }
          }
        }
  } else{
    cl = parallelly::makeClusterPSOCK(n_cores, autoStop = TRUE)
    doParallel::registerDoParallel(cl)
    k = NULL
    regr = foreach::foreach(i = 1:n_steps) %:% # using all cpus
            foreach::foreach(j = 1:n_y) %:% 
              foreach::foreach(k = 1:c) %dopar% {
                # Training function
                train_xgboost = function(X_train, y_train, params, n_estimators=100, seed=666){
                  # Remove observations with missing values in y
                  obs_to_remove = is.na(y_train)
                  X_train_ = X_train[!obs_to_remove,]
                  y_train_ = y_train[!obs_to_remove]
                  set.seed(seed)
                  out = xgboost::xgboost(as.matrix(X_train_), y_train_, params=params, nrounds=n_estimators, verbose=0)
                  return(out)
                }
                train_xgboost(X_train[[i]][mask_y[[j]], ], y_train[mask_y[[j]], k], params, n_estimators=n_estimators, seed=seed)
            }
    #parallel::stopCluster(cl)
    #parallelly::killNode(cl)
    doParallel::stopImplicitCluster()
    parallelly::killNode(cl) # no other choice, it doesnt go away otherwise
    rm(list = "cl")
    gc()
    }
  result = list(regr = regr, X_min = X_min, X_max = X_max, X_min_dummy = X_min_, X_max_dummy = X_max_, cat_indexes=cat_indexes, int_indexes=int_indexes, 
    c=c, b=b,
    beta_0=beta_min, beta_1=beta_max, eps=eps,
    flow=flow,
    bin_indexes=bin_indexes, cat_levels=cat_levels, cat_labels=cat_labels, n_t=n_t, y_uniques=y_uniques, mask_y=mask_y, y_probs=y_probs,
    label_y=label_y, name_y=name_y,
    X1=X)
  class(result) = 'ForestDiffusion'
  return(result)
}

## Below are three functions for cleaning the data after generating/imputing

# unscale the min-max normalization
ForestDiffusion.unscale = function(object, X){
  for (i in 1:NCOL(X)){
    X[,i] = (X[,i] + 1) / 2 # [-1,1] -> [0,1]
    X[,i] = X[,i]*(object$X_max_dummy[i] - object$X_min_dummy[i]) + object$X_min_dummy[i] # [Xmin, Xmax]
  }
  return(X)
}

# Rounding for the categorical variables which are dummy encoded and then remove dummy encoding
ForestDiffusion.clean_dummy_data = function(object, X){
  if (length(object$cat_indexes) > 0){ # ex: [5, 3] and X_names [gender_a gender_b cartype_a cartype_b cartype_c]
    # for all categorical variables, we have prefix ex: ['gender', 'gender']
    prefixes_suffixes = strsplit(names(X),'\\.')
    prefixes = vector("character", NCOL(X))
    for (i in 1:NCOL(X)){
      prefixes[i] = prefixes_suffixes[[i]][1]
    }
    unique_prefixes = unique(prefixes) # uniques prefixes
    for (i in 1:length(unique_prefixes)){
        names_with_prefix = names(X)[grepl(paste0(unique_prefixes[i],'.'), names(X))]
        if (length(names_with_prefix) > 0){
          cat_vars = cbind(rep(0.5, NROW(X)), X[, names_with_prefix])
          max_index = max.col(cat_vars) # argmax of -1, -1, 0 is 0; so as long as they are below 0 we choose the implicit-final class
          X[,names_with_prefix[1]] = max_index # the argmax
          name_no_suffix = strsplit(names(X)[names(X)==names_with_prefix[1]],'\\.')[[1]][1] # ex: cartype.a -> cartype
          names(X)[names(X) == names_with_prefix[1]] = name_no_suffix
          X = X[, !grepl(paste0(unique_prefixes[i],'.'), names(X))] # ex: cartype cartype.b cartype.c; we remove cartype.b cartype.c
        }
    }
  }
  return(X)
}

# Unscale and clip to prevent going beyond min-max and also round of the integers
ForestDiffusion.clip_extremes_clean = function(object, X){
  for (i in object$int_indexes){
    X[,i] = round(X[,i], 0)
  }

  # clip extremes
  for (i in 1:NCOL(X)){
    if (!(i %in% c(object$cat_indexes))){
    small = (X[,i] < object$X_min[i]) + 0.0
    X[,i] = small*object$X_min[i] + (1-small)*X[,i]
    big = (X[,i] > object$X_max[i]) + 0.0
    X[,i] = big*object$X_max[i] + (1-big)*X[,i]
    }
  }

  #For all binary/categorical variable, we must revert to factors with the correct label
  j = 1
  for (i in sort(c(object$bin_indexes, object$cat_indexes))){
    #print(X[,i])
    #print(object$cat_levels[[j]])
    #print(object$cat_labels[[j]])
    X[,i] = factor(X[,i], levels=object$cat_levels[[j]], labels=object$cat_labels[[j]])
    j = j + 1
  }

  # Replace all $&$ back into .
  names(X) = gsub('PlAcEhOlDeR', '\\.', names(X))

  return(X)
}

## The models used in the ODE/SDE

# Return the score-fn or ode-flow output
#' @import xgboost
#' @importFrom stats predict
ForestDiffusion.my_model = function(object, t, y, mask_y){
  b = dim(y)[1]
  c = dim(y)[2]
  X = y

  # Output
  out = X
  out[,] = 0.0
  i = round(t*(object$n_t-1)) + 1 # 1-indexing
  for (j in 1:length(object$y_uniques)){
    if (sum(mask_y[[j]]) > 0){
      for (k in 1:c){
        out[mask_y[[j]], k] = stats::predict(object$regr[[i]][[j]][[k]], as.matrix(X[mask_y[[j]], ]))
      }
    }
  }

  if (!object$flow){
    mu_std = marginal_prob_coef(t, beta_0=object$beta_0, beta_1=object$beta_1)
    out = - out / mu_std[[2]] # score = -z/std
  }
  return(out)
}

#### Below is for Score-based Diffusion Sampling ####

predictor_update_fn = function(score_fn, x, t, h, beta_0=0.1, beta_1=8, flow=FALSE, last_step=FALSE, X_miss=NULL){
  # noise
  z = x
  for (i in 1:NCOL(x)){
    z[,i] = stats::rnorm(NROW(x))
  }

  # Forward SDE
  beta_t = beta_0 + t * (beta_1 - beta_0)
  drift = -0.5 * beta_t * x
  diffusion = sqrt(beta_t)

  # Reverse SDE
  score = score_fn(y=x, t=t)
  drift_reverse = drift - (diffusion^2)*score
  diffusion_reverse = diffusion

  # Euler-Maruyama step
  x_mean = x - drift_reverse * h
  if (!(last_step)){
    x = x_mean + diffusion_reverse*sqrt(h)*z
  } else {
    x = x_mean
  }

  # Replace with ground truth
  if (!is.null(X_miss)){
    y0 = X_miss
    for (i in 1:NCOL(y0)){
      y0[,i] = stats::rnorm(NROW(y0))
    }
    if (last_step){
      X_true = X_miss
    } else{
        if (flow){
        X_true = t*X_miss + (1-t)*y0 # interpolation based on ground-truth for non-missing data
      } else{
        mu_std = marginal_prob_coef(t, beta_0=beta_0, beta_1=beta_1)
        X_true = mu_std[[1]]*X_miss + mu_std[[2]]*y0 # forward step with the SDE
      }
    }
    mask_miss = is.na(X_miss)
    x[!mask_miss] = X_true[!mask_miss] # replace non-missing data with ground truth
  }

  return(x)
}

get_pc_sampler = function(score_fn, denoise=TRUE, N=100, eps=1e-3, beta_0=0.1, beta_1=8, repaint=FALSE, X_miss=NULL){

  pc_sampler = function(prior, r=5, j=5){
    # Initial sample
    x = prior
    timesteps = seq(1, eps, length.out=N)
    h = timesteps - c(timesteps, 0)[2:(N+1)] # true step-size: difference between current time and next time (only the new predictor classes will use h, others will ignore)

    for (i in 1:N){
      x = predictor_update_fn(score_fn, x, timesteps[i], h[i], beta_0=beta_0, beta_1=beta_1, last_step=(i==N), X_miss=X_miss)
    }

    #if (denoise){ # Tweedie formula
    #  std = marginal_prob_coef(eps, beta_0=beta_0, beta_1=beta_1)[[2]]
    #  x = x + (std ** 2)*score_fn(y=x, t=eps)
    #}

    return(x)
  }

  pc_sampler_repaint = function(prior, r=5, j=5){
    # Initial sample
    x = prior
    timesteps = seq(1, eps, length.out=N)
    h = timesteps - c(timesteps, 0)[2:(N+1)] # true step-size: difference between current time and next time (only the new predictor classes will use h, others will ignore)

    i_repaint = 0
    i = 1
    while (i <= N){
      x = predictor_update_fn(score_fn, x, timesteps[i], h[i], beta_0=beta_0, beta_1=beta_1, last_step=(i==N), X_miss=X_miss)
      if (i_repaint < r-1 & i %% j == 0){ # we did j iterations, but not enough repaint, we must repaint again
        
        # Noise
        z = x
        for (i_ in 1:NCOL(x)){
          z[,i_] = stats::rnorm(NROW(x))
        }

        # Forward SDE
        beta_t = beta_0 + timesteps[i] * (beta_1 - beta_0)
        drift = -0.5 * beta_t * x
        diffusion = sqrt(beta_t)

        # Going forward in time
        h_ = sum(h[(i-j+1):i])
        x_mean = x + drift * h_
        x = x_mean + diffusion*sqrt(h_)*z

        # iterate back
        i_repaint = i_repaint + 1
        i = i - j
      }
      else{
        if (i_repaint == r-1 & i %% j == 0){ # we did j iterations and enough repaint, we continue and reset the repaint counter
          i_repaint = 0
        }
      }
      i = i + 1
    }

    #if (denoise){ # Tweedie formula
    #  std = marginal_prob_coef(x, eps, beta_0=beta_0, beta_1=beta_1)[[2]]
    #  x = x + (std ** 2)*score_fn(y=x, t=eps)
    #}
    return(x)
  }

  if (repaint) return(pc_sampler_repaint)
  else return(pc_sampler)
}


#### Below is for Flow-Matching Sampling ####

# Euler solver
euler_solve = function(y0, my_model, N=101){
  h = 1 / (N-1)
  y = y0
  t = 0
  for (i in 1:(N-1)){
    y = y + h*my_model(t=t, y=y)
    t = t + h
  }
  return(y)
}

#' @title Generate new observations with a trained ForestDiffusion model
#' @description Generate new observations by solving the reverse SDE (vp) / ODE (flow) starting from pure Gaussian noise.
#' @param object a ForestDiffusion object
#' @param batch_size (optional) number of observations generated; if not provided, will generate as many observations as the original dataset
#' @param n_t (optional) number of noise levels (and sampling steps); increase for higher performance, but slows down training and sampling; if not provided, will use the same n_t as used in training.
#' @return Returns a data.frame with the generated data
#' @param seed (optional) random seed used
#' @examples
#'  \dontrun{
#'  data(iris)
#'  X = data.frame(iris[,1:4])
#'  y = iris[,5]
#'  
#'  ## Generation
#'  
#'  Xy = X
#'  Xy$y = y
#'  forest_model = ForestDiffusion(X=Xy, n_cores=1, n_t=50, duplicate_K=50, flow=TRUE)
#'  Xy_fake = ForestDiffusion.generate(forest_model, batch_size=NROW(Xy))
#' }  
#'
#' @references 
#' Alexia Jolicoeur-Martineau, Kilian Fatras, Tal Kachman. 
#' Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees. 
#' arXiv:2309.09968.
#' @export

ForestDiffusion.generate = function(object, batch_size=NULL, n_t=NULL, seed=NULL){

  if (!is.null(seed)) set.seed(seed)
  if (is.null(batch_size)) batch_size = object$b
  if (is.null(n_t)) n_t = object$n_t

  # Generate prior noise
  y0 = data.frame(matrix(vector(), batch_size, object$c))
  for (i in 1:(object$c)){
    y0[,i] = stats::rnorm(batch_size)
  }
  names(y0) = names(object$X1)

  # Generate random labels
  label_y = object$y_uniques[max.col(t(stats::rmultinom(batch_size, 1, object$y_probs)))]
  mask_y = vector("list", length(object$y_uniques)) # mask for which observations has a specific value of y
  for (i in 1:length(object$y_uniques)){
    mask_y[[i]] = rep(FALSE, batch_size)
    mask_y[[i]][label_y == object$y_uniques[i]] = TRUE
  }
  my_model_fn = partial(ForestDiffusion.my_model, object=object, mask_y=mask_y)

  if (object$flow){
    solution = euler_solve(my_model=my_model_fn, y0=y0, N=n_t)
  } else{
    solution = get_pc_sampler(my_model_fn, denoise=TRUE, N=object$n_t, eps=object$eps, 
      beta_0=object$beta_0, beta_1=object$beta_1, repaint=FALSE)(y0)
  }
  solution = ForestDiffusion.unscale(object, solution)
  solution = ForestDiffusion.clean_dummy_data(object, solution)
  solution = ForestDiffusion.clip_extremes_clean(object, solution)

  # Concatenate y label if needed
  if (!is.null(object$label_y)) solution[object$name_y] = label_y

  return (solution)
}

#' @title Impute missing data with a trained ForestDiffusion model
#' @description Impute missing data by solving the reverse SDE while keeping the non-missing data intact.
#' @param object a ForestDiffusion object
#' @param k number of imputations 
#' @param X (optional) data.frame of the dataset to be imputed; If not provided, the training dataset will be imputed instead
#' @param label_y (optional) vector containing the outcome variable if it is categorical for improved performance by training separate models per class; cannot contain missing values; if not provided, the training label_y will be used if it exists.
#' @param repaint If TRUE, it will impute using the REPAINT technique for improved performance
#' @param r number of repaints (default=10)
#' @param j jump size in percentage (default: 10 percent of the samples), this is part of REPAINT
#' @param n_t (optional) number of noise levels (and sampling steps); increase for higher performance, but slows down training and sampling; if not provided, will use the same n_t as used in training.
#' @param seed (optional) random seed used
#' @return Returns a data.frame with the generated data
#' @examples
#'  \dontrun{
#'  data(iris)
#'  X = data.frame(iris[,1:4])
#'  y = iris[,5]
#'  
#'  ## Imputation
#'  
#'  # add missing data
#'  Xy = missForest::prodNA(Xy, noNA = 0.2)
#'  
#'  nimp = 5 # number of imputations needed
#'  Xy = X
#'  Xy$y = y
#'  forest_model = ForestDiffusion(X=Xy, n_cores=1, n_t=50, duplicate_K=50, flow=FALSE)
#'  # regular (fast)
#'  Xy_fake = ForestDiffusion.impute(forest_model, k=nimp)
#'  # REPAINT (slow, but better)
#'  Xy_fake = ForestDiffusion.impute(forest_model, repaint=TRUE, r=10, j=5, k=nimp)
#' }
#'
#' @references
#' Alexia Jolicoeur-Martineau, Kilian Fatras, Tal Kachman. 
#' Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees. 
#' arXiv:2309.09968.
#' 
#' Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool. 
#' RePaint: Inpainting using Denoising Diffusion Probabilistic Models. 
#' arXiv:2201.09865.
#' @export

ForestDiffusion.impute = function(object, k=1, X=NULL, label_y=NULL, repaint=FALSE, r=5, j=0.1, n_t=NULL, seed=NULL){ # X is data with missing values
  if (object$flow == 'flow') stop("Cannot use imputation with flow=TRUE, please retrain with flow=FALSE")
  if (sum(is.na(object$X1)>0) == 0) stop("Cannot imputate when data has no missing values")

  if (!is.null(seed)) set.seed(seed)

  if (is.null(X)) X = object$X1
  if (is.null(label_y)) label_y = object$label_y
  if (is.null(n_t)) n_t = object$n_t

  if (is.null(label_y)){ # single category 0
    mask_y = vector("list", 1)
    mask_y[[1]] = rep(TRUE, NROW(X))
  } else{
    mask_y = vector("list", length(object$y_uniques)) # mask for which observations has a specific value of y
    for (i in 1:length(object$y_uniques)){
      mask_y[[i]] = rep(FALSE, NROW(X))
      mask_y[[i]][label_y == object$y_uniques[i]] = TRUE
    }
  }

  my_model_imputation = partial(ForestDiffusion.my_model, object=object, mask_y=mask_y)

  imputed_data = vector("list", k)
  for (i in 1:k){

    # Generate prior noise
    y0 = data.frame(matrix(vector(), NROW(X), NCOL(X)))
    for (i_ in 1:(NCOL(X))){
      y0[,i_] = stats::rnorm(NROW(X))
    }
    names(y0) = names(X)

    solution = get_pc_sampler(my_model_imputation, denoise=TRUE, N=object$n_t, eps=object$eps, 
      beta_0=object$beta_0, beta_1=object$beta_1, repaint=repaint, X_miss=X)(y0, r=r, j=round(j*n_t))
    solution = ForestDiffusion.unscale(object, solution)
    solution = ForestDiffusion.clean_dummy_data(object, solution)
    solution = ForestDiffusion.clip_extremes_clean(object, solution)
    # Concatenate y label if needed
    if (!is.null(object$label_y)) solution[object$name_y] = label_y
    imputed_data[[i]] = solution
  }
  return(imputed_data)
}


#' @title Evaluate an expression in multiple generated/imputed datasets
#' @description It performs a computation for each dataset (function modified from mice::with.mids). For example, you can use this function to train a different glm model per dataset and then pool the estimates (akin to with multiple imputations, but more general so that it can be applied to any dataset).
#' @param data a list of datasets
#' @param expr An expression to evaluate for each imputed data set. Formula's
#' containing a dot (notation for "all other variables") do not work.
#' @return An object of S3 class mira
#' @examples
#' \dontrun{
#' library(mice)
#' 
#' # Load iris
#' data(iris)
#' Xy = data.frame(iris[,1:4])
#' Xy$y = iris[,5]
#' 
#' # add missing data
#' Xy = missForest::prodNA(Xy, noNA = 0.2)
#' 
#' forest_model = ForestDiffusion(X=Xy, n_cores=1, n_t=50, duplicate_K=50, flow=FALSE)
#' nimp = 5 # number of imputations needed
#' # regular (fast)
#' Xy_imp = ForestDiffusion.impute(forest_model, k=nimp)
#' # REPAINT (slow, but better)
#' Xy_imp = ForestDiffusion.impute(forest_model, repaint=TRUE, r=10, j=5, k=nimp)
#' 
#' # Fit a model per imputed dataset
#' fits <- with_datasets(Xy_imp, glm(y ~ Sepal.Length, family = 'binomial'))
#' 
#' # Pool the results
#' mice::pool(fits) 
#' }
#' 
#' @export

with_datasets <- function(data, expr) {
  call <- match.call()
  analyses <- as.list(seq_len(length(data)))

  # do the repeated analysis, store the result.
  for (i in seq_along(analyses)) {
    data.i <- data[[i]]
    analyses[[i]] <- eval(expr = substitute(expr), envir = data.i, enclos = parent.frame())
    if (is.expression(analyses[[i]])) {
      analyses[[i]] <- eval(expr = analyses[[i]], envir = data.i, enclos = parent.frame())
    }
  }

  # return the complete data analyses as a list of length nimp
  object <- list(call = call, call1 = data$call, nmis = data$nmis, analyses = analyses)
  # formula=formula(analyses[[1]]$terms))
  oldClass(object) <- c("mira", "matrix")
  object
}
