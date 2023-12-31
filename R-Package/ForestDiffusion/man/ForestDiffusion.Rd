% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ForestDiffusion.R
\name{ForestDiffusion}
\alias{ForestDiffusion}
\title{Diffusion and Flow-based XGBoost Model for generating or imputing data}
\usage{
ForestDiffusion(
  X,
  n_cores,
  label_y = NULL,
  name_y = "y",
  n_t = 50,
  flow = TRUE,
  max_depth = 7,
  n_estimators = 100,
  eta = 0.3,
  duplicate_K = 50,
  true_min_max_values = NULL,
  eps = 0.001,
  beta_min = 0.1,
  beta_max = 8,
  seed = NULL
)
}
\arguments{
\item{X}{data.frame of the dataset to be used.}

\item{n_cores}{number of cpu cores used (if NULL, it will use all cores, otherwise it will use min(n_cores, max_available_cores); using more cores makes training faster, but increases the memory cost (so reduce it if you have memory problems)}

\item{label_y}{optional vector containing the outcome variable if it is categorical for improved performance by training separate models per class; cannot contain missing values}

\item{name_y}{name of label_y}

\item{n_t}{number of noise levels (and sampling steps); increase for higher performance, but slows down training and sampling}

\item{flow}{If TRUE, uses flow (an ODE deterministic method); otherwise uses vp (a SDE stochastic method); 'vp' generally has slightly worse performance, but it is the only method that can be used for imputation}

\item{max_depth}{max depth of the trees per XGBoost model}

\item{n_estimators}{number of trees per XGBoost model}

\item{eta}{learning rate per XGBoost model}

\item{duplicate_K}{number of noise per sample (or equivalently the number of times the rows of the dataset are duplicated); should be as high as possible; higher values increase the memory demand}

\item{true_min_max_values}{(optional) list of form [[min_x, min_y], [max_x, max_y]]; If  provided, we use these values as the min/max for each variables when using clipping}

\item{eps}{minimum noise level}

\item{beta_min}{value of the beta_min in the vp process}

\item{beta_max}{value of the beta_max in the vp process}

\item{seed}{(optional) random seed used}
}
\value{
Returns an object of the class "ForestDiffusion" which is list containing the XGBoost model fits
}
\description{
Train XGBoost regression models to estimate the score-function (for diffusion models) or the flow (flow-based models). These models can then be used to generate new fake samples or impute missing values.
}
\examples{
 \dontrun{
 data(iris)
 iris[,1:4] = missForest::prodNA(iris[,1:4], noNA = 0.2) # adding missing data
 X = data.frame(iris[,1:4])
 y = iris[,5]
 
 ## Generation
 
 # Classification problem (outcome is categorical)
 forest_model = ForestDiffusion(X=X, n_cores=1, label_y=y, n_t=50, duplicate_K=50, flow=TRUE)
 # last variable will be the label_y
 Xy_fake = ForestDiffusion.generate(forest_model, batch_size=NROW(iris))
 
 # When you do not want to train a seperate model per model (or you have a regression problem)
 Xy = X
 Xy$y = y
 forest_model = ForestDiffusion(X=Xy, n_cores=1, n_t=50, duplicate_K=50, flow=TRUE)
 Xy_fake = ForestDiffusion.generate(forest_model, batch_size=NROW(iris))
 
 ## Imputation
 
 # flow=TRUE generate better data but it cannot impute data
 forest_model = ForestDiffusion(X=Xy, n_cores=1, n_t=50, duplicate_K=50, flow=FALSE)
 nimp = 5 # number of imputations needed
 # regular (fast)
 Xy_fake = ForestDiffusion.impute(forest_model, k=nimp)
 # REPAINT (slow, but better)
 Xy_fake = ForestDiffusion.impute(forest_model, repaint=TRUE, r=10, j=5, k=nimp)
}
 
}
\references{
Alexia Jolicoeur-Martineau, Kilian Fatras, Tal Kachman. 
Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees. 
arXiv:2309.09968.
}
