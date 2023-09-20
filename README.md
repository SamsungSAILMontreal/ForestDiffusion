# Tabular_Flow_Matching

This repo contains the official implementation of the paper [Generating and Imputing Tabular Data via Diffusion and Flow-based XGBoost Models](https://arxiv.org/abs/2309.09968). To make it easily accessible, we release our code through a Python library and an R package. See also our [blog post](https://ajolicoeur.wordpress.com/2023/09/19/xgboost-diffusion/) for more information.

![](https://raw.githubusercontent.com/AlexiaJM/Tabular_Flow_Matching/master/iris_cropped.png)

## R library

The code for the R package is in the R-package folder. You can install the R package in R using:
```
install.packages('ForestDiffusion')
```

For anything R related, please see XXXXX for the R documentation and vignette explaining how to use the package. The rest of the README is specific to the Python library.

## Install the Python library

The code for the Python library is in the Python-Package folder. To install the latest stable version on Python, run in bash :

```
pip install ForestDiffusion
```

## To run the experiments in the paper

Install the required packages:

```
pip install ForestDiffusion
pip install -r requirements.txt
```

Then run script_generation and script_imputation to replicate the analyses:

```
export out_path='/home/mila/j/jolicoea/where_to_store_results.txt'

# Generation with no missing data
myargs=" --methods oracle GaussianCopula TVAE CTABGAN CTGAN stasy TabDDPM forest_diffusion --diffusion_type flow --out_path ${out_path}"
python script_generation.py ${myargs}
myargs=" --methods forest_diffusion --diffusion_type vp --out_path ${out_path}"
python script_generation.py ${myargs}

# Generation with missing data
myargs=" --methods oracle GaussianCopula TVAE CTABGAN CTGAN stasy TabDDPM --add_missing_data True --imputation_method MissForest --out_path ${out_path}"
python script_generation.py ${myargs}
myargs=" --methods forest_diffusion --diffusion_type flow --add_missing_data True --imputation_method none --out_path ${out_path}"
python script_generation.py ${myargs}
myargs=" --methods forest_diffusion --diffusion_type vp --add_missing_data True --imputation_method none --out_path ${out_path}"
python script_generation.py ${myargs}

# Imputation with missing data
myargs=" --methods oracle KNN miceforest ice softimpute OT GAIN forest_diffusion --diffusion_type vp --out_path ${out_path}"
python script_imputation.py ${myargs}
```

As always, note that from the code refactoring and cleaning, the randomness will be different, so expect to get slightly different numbers than the exact ones from the paper.

## For regular usage on your dataset

Install the pip package (as shown above). 

Examples to generate new samples given your dataset (can contain missing values):

```
# Classification problem (outcome is categorical)
forest_model = ForestDiffusion(X, y_label=y, n_t=50, duplicate_K=100, bin_indexes=[3], cat_indexes=[0,5], int_indexes=[1,2], diffusion_type='flow', n_jobs=-1)
Xy_fake = forest_model.generate(batch_size=X.shape[0]) # last variable will be the y_label

# Regression problem (outcome is continuous)
Xy = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
forest_model = ForestDiffusion(Xy, n_t=50, duplicate_K=100, bin_indexes=[2], cat_indexes=[0,1], int_indexes=[], diffusion_type='flow', n_jobs=-1)
Xy_fake = forest_model.generate(batch_size=X.shape[0])
```

Examples to impute your dataset:

```
nimp = 5 # number of imputations needed
Xy = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
forest_model = ForestDiffusion(Xy, n_t=50, duplicate_K=100, bin_indexes=[4], cat_indexes=[1,2], int_indexes=[0], diffusion_type='vp', n_jobs=-1)
Xy_fake = forest_model.impute(k=nimp) # regular (fast)
Xy_fake = forest_model.impute(repaint=True, r=10, j=5, k=nimp) # REPAINT (slow, but better)
```

## Hyperparameters

We list the important hyperparameters below, their default values, and how to tune them:

```
n_jobs = -1 # number of cpus/processes used for the parallel loop (-1 means all cpus; using a small number like n_jobs=4 will reduce training speed, but reduce memory load)
y_label = None # provide the outcome variable if it is categorical for improved performance by training separate models per class; cannot contain missing values
bin_indexes = [] # vector that indicates which column is binary
cat_indexes = [] # vector that indicates which column is categorical (>=3 categories)
int_indexes = [] # vector that indicates which column is an integer (ordinal variables such as the number of cats in a box)
n_t = 50 # number of noise levels (and sampling steps); increase for higher performance, but slows down training and sampling
diffusion_type = 'flow' # type of process (flow = ODE, vp = SDE); vp generally has slightly worse performance, but it is the only method that can be used for imputation
duplicate_K = 100 # number of noise per sample (or equivalently the number of times the rows of the dataset are duplicated); should be as high as possible; higher values increase the memory demand
seed = 666 # random seed value
max_depth = 7 # max depth of the tree; recommended to leave at default
n_estimators = 100 # number of trees per XGBoost model; recommended to leave at default
```

Regarding the imputation with REPAINT, there are two important hyperparameters:
```
r = 10 # number of repaints, 5 or 10 is good
j = 5 # jump size; should be around 10% of n_t
```

## Potential memory problems and solutions 😭

Our method trains p\*n_t models in parallel using CPUs, where p is the number of variables and n_t is the number of noise levels. Furthermore, we make the dataset much bigger by duplicating the rows many times (100 times is the default).

To speed up the training, you will need as many CPUs as possible. Training the multiple models using only 4 CPUs could take a long time. However, the more CPUs you use, the higher the memory cost will be! This is because each worker/CPU will train its own model, which will require its own amount of memory (RAM). So, there is a balance to be reached between enough CPUs for speed but not too much so that it doesn't blow up the memory.

If you use clusters, you can control the amount of CPUs and RAM you take. There are also ways to force Python to only use a subset of the CPUs (using the n_jobs parameters). We provide below some hyperparameters that can be changed to reduce the memory load:
```
duplicate_K = 100 # lowering this value will reduce memory demand and possibly performance (memory is proportional to this value)
n_jobs = -1 # number of cpus/processes used for the parallel loop (-1 means all cpus; using a small number like n_jobs=4 will reduce training speed, but reduce memory load)
n_t = 50 # reducing this value will likely reduce memory demand and possibly performance (stay at n_t=50 or higher)
y_label = None # using None will reduce memory demand (since using this will train n_classes times more models)
max_depth = 7 # reducing the depth of trees will reduce memory demand
n_estimators = 100 # reducing the number of trees will reduce memory demand
```

## References

If you find the code useful, please consider citing
```bib
@misc{jolicoeurmartineau2023generating,
      title={Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees}, 
      author={Alexia Jolicoeur-Martineau and Kilian Fatras and Tal Kachman},
      year={2023},
      eprint={2309.09968},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
