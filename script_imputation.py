#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn

import ot as pot

import time
import os
import pickle as pkl
import copy

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from utils import *
from softimpute import softimpute, cv_softimpute
from data_loaders import dataset_loader
from imputers import OTimputer
from sklearn.model_selection import train_test_split
from metrics import test_on_multiple_models, test_imputation_regression

from ForestDiffusion import ForestDiffusionModel
from sklearn.impute import KNNImputer
import gain
import argparse
import miceforest as mf
from missforest import MissForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str, default='jolicoea/tabular_imputation_results.txt',
                    help='filename for the results')

parser.add_argument("--restore_from_name", type=str2bool, default=False, help="if True, restore session based on name")
parser.add_argument("--name", type=str, default='my_exp', help="used when restoring from crashed instances")

parser.add_argument("--methods", type=str, nargs='+', default=['oracle', 'GAIN', 'KNN', 'MissForest', 'miceforest', 'forest_diffusion', 'ice', 'softimpute', 'OT'], help="oracle, GAIN, KNN, MissForest, miceforest, forest_diffusion, ice, softimpute, OT")
parser.add_argument('--nexp', type=int, default=3,
                    help='number of experiences per method')
parser.add_argument('--nimp', type=int, default=5,
                    help='number of imputations per method')
parser.add_argument('--n_tries', type=int, default=5,
                    help='number of models trained with different seeds in the metrics')

parser.add_argument('--datasets', nargs='+', type=str, default=['iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'california', 'bean', 'tictactoe','congress','car'],
                    help='datasets on which to run the experiments')
parser.add_argument('--naug', type=int, default=5,
                    help='number of augmentation if used')
parser.add_argument('--p', type=float, default=0.2, help='Proportion of imps')
parser.add_argument('--n_neighbors', type=int, default=1, help='n_neighbors')

# Forest hyperparameters
parser.add_argument('--forest_model', type=str, default='xgboost', help='xgboost, random_forest, lgbm, catboost')
parser.add_argument('--diffusion_type', type=str, default='vp', help='flow (flow-matching), vp (Variance-Preserving diffusion)')
parser.add_argument('--n_t', type=int, default=50, help='number of times t in [0,1]')
parser.add_argument('--n_t_sampling', type=int, default=0, help='number of times t in [0,1] for sampling (0 will uses n_t steps; ignore this parameter honestly, its worth changing)')
parser.add_argument('--max_depth', type=int, default=7, help='max tree depth (xgboost, random_forest)')
parser.add_argument('--num_leaves', type=int, default=31, help='max number of leaves (lgbm)')
parser.add_argument('--n_estimators', type=int, default=100, help='number of trees (xgboost, random_forest, lgbm)')
parser.add_argument('--eta', type=float, default=0.3, help='lr (xgboost, random_forest, lgbm)')
parser.add_argument('--duplicate_K', type=int, default=100, help='number of times to duplicate the data for improved performanced of forests')
parser.add_argument('--gpu_hist', type=str2bool, default=False, help='If True, xgboost use the GPU')
parser.add_argument('--ycond', type=str2bool, default=True, help='If True, make a different forest model per label (when its not regression obviously)')
parser.add_argument('--eps', type=float, default=1e-3, help='')
parser.add_argument('--beta_min', type=float, default=0.1, help='')
parser.add_argument('--beta_max', type=float, default=8, help='')
parser.add_argument('--repaint_r', type=int, default=10, help='number of repaints')
parser.add_argument('--repaint_j', type=float, default=0.1, help='percentage jump size of repaint (jump size=5 make sense for n_t=50)')
parser.add_argument('--n_jobs', type=int, default=-1, help='')
parser.add_argument('--n_batch', type=int, default=1, help='If >0 use the data iterator with the specified number of batches')

args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":

    OTLIM = 5000

    if args.n_t_sampling == 0:
        args.n_t_sampling = args.n_t

    if 'forest_diffusion' in args.methods and args.diffusion_type != 'flow': # we want the forest_diffusion results, but also the ones with repaint
        args.methods.append('forest_diffusion_repaint')

    dataset_index_start = 0
    method_index_start = 0
    if args.restore_from_name:
        if os.path.isfile(args.name):
            with open(args.name, 'r') as f: # where we track where we are, to restore sessions after crashes
                dataset_index_start, method_index_start = f.read().split('&')
                dataset_index_start = int(dataset_index_start)
                method_index_start = int(method_index_start)

    for dataset_index in range(dataset_index_start, len(args.datasets)):

        dataset = args.datasets[dataset_index]
        print(dataset)

        X, bin_x, cat_x, int_x, y, bin_y, cat_y, int_y = dataset_loader(dataset)
     
        # Binary
        bin_indexes = []
        if bin_x is not None:
            bin_indexes = bin_indexes + bin_x
        bin_indexes_no_y = copy.deepcopy(bin_indexes)
        if bin_y:
            bin_indexes.append(X.shape[1])

        # Categorical (>=2 classes)
        cat_indexes = []
        if cat_x is not None:
            cat_indexes = cat_indexes + cat_x
        cat_indexes_no_y = copy.deepcopy(cat_indexes)
        if cat_y:
            cat_indexes.append(X.shape[1])

        not_cat_indexes = [i for i in range(X.shape[1]+1) if i not in cat_indexes]

        # Integers
        int_indexes = []
        if int_x is not None:
            int_indexes = int_indexes + int_x
        int_indexes_no_y = copy.deepcopy(int_indexes)
        if int_y:
            int_indexes.append(X.shape[1])

        score_mae_min = {}
        score_mae_avg = {}
        score_W1_miss = {}
        score_W1_train = {}
        score_W1_test = {}
        mean_var = {}
        mean_mad_mean = {}
        mean_mad_median = {}
        percent_bias = {}
        coverage_rate = {}
        AW = {}
        time_taken = {}
        for method in args.methods:
            score_mae_min[method] = 0.0
            score_mae_avg[method] = 0.0
            score_W1_miss[method] = 0.0
            score_W1_train[method] = 0.0
            score_W1_test[method] = 0.0
            mean_var[method] = 0.0
            mean_mad_mean[method] = 0.0
            mean_mad_median[method] = 0.0
            percent_bias[method] = 0.0
            coverage_rate[method] = 0.0
            AW[method] = 0.0
            time_taken[method] = 0.0

        R2 = {}
        f1 = {}
        for method in args.methods:
            R2[method] = {}
            f1[method] = {}
            for test_type in ['mean','lin','linboost', 'tree', 'treeboost']:
                R2[method][test_type] = 0.0
                f1[method][test_type] = 0.0

        coverage = {}
        for method in args.methods:
            coverage[method] = 0.0

        p = args.p
        data = {"imp": {}, "forest_models": [], "forest_models_time": []}

        for method_index in range(method_index_start, len(args.methods)):
            
            method = args.methods[method_index]
            print(f'method={method}')

            with open(args.name, 'w') as f: # where we track where we are, to restore sessions after crashes
                f.write(f'{dataset_index}&{method_index}')

            for n in range(args.nexp):

                print(f'nexp={n}/{args.nexp-1}')

                torch.manual_seed(n)
                np.random.seed(n)

                # Need to train/test split for evaluating the linear regression performance and for W1 based on test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n, stratify=y if bin_y or cat_y else None)
                Xy_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
                Xy_test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)

                ### Each entry from the second axis has a probability p of being NA 
                X_true = torch.tensor(X_train)
                mask_x = (torch.rand(X_true.shape) < p).to(device=device)

                # we must remove observations with only missing data
                obs_to_remove = mask_x.bool().all(axis=1)
                mask_x = mask_x[~obs_to_remove]
                X_train = X_train[~obs_to_remove.detach().cpu().numpy()]
                X_true = X_true[~obs_to_remove]
                y_train = y_train[~obs_to_remove.detach().cpu().numpy()]
                Xy_train = Xy_train[~obs_to_remove.detach().cpu().numpy()]

                # Now adding the outcome Y without missing data
                Xy_true = torch.tensor(Xy_train)
                mask = torch.zeros_like(Xy_true)
                mask[:, :-1] = mask_x

                Xy_nas = Xy_true.clone()
                Xy_nas[mask.bool()] = np.nan # torch data
                data_nas = Xy_nas.cpu().numpy() # numpy data
                M = mask.sum(1) > 0

                mask_np = mask.detach().cpu().numpy()
                M_np = M.detach().cpu().numpy()
                data["imp"][method] = []

                start = time.time()

                if method == 'oracle':
                    for imp_i in range(args.nimp):
                        data["imp"][method].append(Xy_train)

                elif method == 'GAIN':
                    gain_parameters = {'batch_size': 128,
                                     'hint_rate': 0.9,
                                     'alpha': 100,
                                     'iterations': 10000,
                                     'nimp': args.nimp,
                                     'cat_indexes': bin_indexes + cat_indexes}
                    my_imp = gain.gain(data_nas, gain_parameters)
                    for imp_i in range(args.nimp):
                        data["imp"][method].append(my_imp[imp_i])

                # Note: For KNN its pointless to for-loop over k imputations, there is a single possible solution, so we copy it k times

                elif method == 'KNN': # KNN with z-score (z-score the categorical which is not ideal, but works fine)

                    scaler = StandardScaler()
                    data_nas_std = scaler.fit_transform(data_nas)

                    # Use KNN Imputation to match observations with missing values to closest fake ones
                    X_KNN = KNNImputer(n_neighbors=args.n_neighbors, weights="uniform").fit_transform(data_nas_std)

                    # Normalized back to normal range
                    my_imp = scaler.inverse_transform(X_KNN)
                    my_imp = np.round(my_imp, decimals=8) # hack needed for categorical variables

                    for imp_i in range(args.nimp):
                        data["imp"][method].append(my_imp)

                elif method == 'MissForest':
                    data_nas_ = copy.deepcopy(data_nas)
                    for imp_i in range(args.nimp):
                        imputer = MissForest(random_state=imp_i)
                        my_imp = imputer.fit_transform(data_nas_, cat_vars=bin_indexes + cat_indexes if len(bin_indexes + cat_indexes) > 0 else None)
                        data["imp"][method].append(my_imp)

                elif method == 'miceforest':
                    # Convert to Pandas
                    data_pd = pd.DataFrame(data_nas, columns = [str(i) for i in range(data_nas.shape[1])])
                    # indicate which column is categorical so that they are handled properly
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category') 
                    kds = mf.ImputationKernel(data_pd, save_all_iterations=False, datasets=args.nimp, random_state=n)
                    kds.mice(5) # 5 iterations is the default and should be enough
                    for imp_i in range(args.nimp):
                        my_imp = kds.complete_data(dataset=imp_i).to_numpy()
                        data["imp"][method].append(my_imp)

                elif method == 'forest_diffusion':

                    if args.ycond and (bin_y or cat_y):
                        forest_model = ForestDiffusionModel(X=data_nas[:,:-1], 
                            label_y=data_nas[:,-1],
                            n_t=args.n_t,
                            model=args.forest_model, # in random_forest, xgboost, lgbm
                            diffusion_type=args.diffusion_type, # vp, flow
                            max_depth = args.max_depth, n_estimators = args.n_estimators, # random_forest and xgboost hyperparameters
                            eta=args.eta, # xgboost hyperparameters
                            num_leaves=args.num_leaves, # lgbm hyperparameters
                            gpu_hist=args.gpu_hist,
                            duplicate_K=args.duplicate_K,
                            cat_indexes=cat_indexes_no_y,
                            bin_indexes=bin_indexes_no_y,
                            int_indexes=int_indexes_no_y,
                            n_jobs=args.n_jobs,
                            n_batch=args.n_batch,
                            eps=args.eps, beta_min=args.beta_min, beta_max=args.beta_max,
                            seed=n)
                    else:
                        forest_model = ForestDiffusionModel(X=data_nas, 
                            n_t=args.n_t,
                            model=args.forest_model, # in random_forest, xgboost, lgbm
                            diffusion_type=args.diffusion_type, # vp, flow
                            max_depth = args.max_depth, n_estimators = args.n_estimators, # random_forest and xgboost hyperparameters
                            eta=args.eta, # xgboost hyperparameters
                            num_leaves=args.num_leaves, # lgbm hyperparameters
                            gpu_hist=args.gpu_hist,
                            duplicate_K=args.duplicate_K,
                            cat_indexes=cat_indexes,
                            bin_indexes=bin_indexes,
                            int_indexes=int_indexes,
                            n_jobs=args.n_jobs,
                            n_batch=args.n_batch,
                            eps=args.eps, beta_min=args.beta_min, beta_max=args.beta_max,
                            seed=n)

                    data["forest_models"].append(forest_model)
                    data["forest_models_time"].append(time.time() - start)
                    assert args.diffusion_type == 'vp'
                    my_imp = forest_model.impute(k=args.nimp, n_t=args.n_t_sampling)
                    for imp_i in range(args.nimp):
                        data["imp"][method].append(my_imp[imp_i] if args.nimp > 1 else my_imp)


                elif method == 'forest_diffusion_repaint':
                    forest_model = data["forest_models"][n] # already trained
                    my_imp = forest_model.impute(repaint=True, r=args.repaint_r, j=args.repaint_j, k=args.nimp, n_t=args.n_t_sampling)
                    for imp_i in range(args.nimp):
                        data["imp"][method].append(my_imp[imp_i] if args.nimp > 1 else my_imp)

                elif method == 'ice':
                    for imp_i in range(args.nimp):
                        ice_mean = IterativeImputer(random_state=imp_i, max_iter=10, sample_posterior=True) #, min_value=np.nanmin(data_nas, axis=0), max_value=np.nanmax(data_nas, axis=0))
                        ice_mean.fit(data_nas)
                        my_imp = ice_mean.transform(data_nas)
                        my_imp = clip_extremes(data_nas, my_imp, int_indexes=int_indexes+bin_indexes+cat_indexes) # Alexia: added to reduce the large variance in this method
                        data["imp"][method].append(my_imp)

                elif method == 'softimpute':
                    cv_error, grid_lambda = cv_softimpute(data_nas, grid_len=15)
                    lbda = grid_lambda[np.argmin(cv_error)]
                    for imp_i in range(args.nimp):
                        my_imp = softimpute((data_nas), lbda)[1]
                        my_imp = clip_extremes(data_nas, my_imp, int_indexes=int_indexes+bin_indexes+cat_indexes) # Alexia: added to reduce the large variance in this method
                        data["imp"][method].append(my_imp)

                elif method == 'OT':
                    epsilon = pick_epsilon(Xy_nas, 0.5, 0.05)
                    for imp_i in range(args.nimp):
                        sk_imputer = OTimputer(eps=epsilon, niter=3000, batchsize=128, lr=1e-2)
                        sk_imp = sk_imputer.fit_transform(Xy_nas.clone(), report_interval=500,
                                                         verbose=False, X_true=None)
                        my_imp = sk_imp.detach().cpu().numpy()
                        my_imp = clip_extremes(data_nas, my_imp, int_indexes=int_indexes+bin_indexes+cat_indexes) # Alexia: added to reduce the large variance in this method
                        data["imp"][method].append(my_imp)

                end = time.time()
                if method == 'forest_diffusion_repaint':
                    time_taken[method] += (end - start + data["forest_models_time"][n]) / args.nexp # adding the time it took to train the model
                else:
                    time_taken[method] += (end - start) / args.nexp

                #print(Xy_train)
                #print(np.array2string(Xy_fake, formatter={'float_kind':'{0:.1f}'.format}))

                # Mixed data is tricky, RMSE, nearest neighboors (for the coverage) and Wasserstein distance (based on L2) are not scale invariant
                # To ensure that the scaling between variables is relatively uniformized, we take inspiration from the Gower distance used in mixed-data KNNs: https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b
                # Continuous: we do min-max normalization (to use Gower |x1-x2|/(max-min) as distance)
                # Categorical: We one-hot and then divide by 2 (e.g., 0 0 0.5 with 0.5 0 0 will have distance 0.5 + 0.5 = 1)
                # After these transformations, taking the L1 (City-block / Manhattan distance) norm distance will give the Gower distance
                Xy_train_scaled, Xy_test_scaled, mask_scaled, _, df_names_before, df_names_after = minmax_scale_dummy(Xy_train, Xy_test, cat_indexes, mask=mask_np, divide_by=2)
                Xy_train_minmax, Xy_test_minmax, _ = minmax_scale(Xy_train, Xy_test, cat_indexes)

                M_scaled = np.sum(mask_scaled, axis=1) > 0
                n_miss=mask_np.astype(bool).sum()

                # concatenate the fakes
                Xy_fake_scaled = [None for _ in range(args.nimp)]
                Xy_fake = [None for _ in range(args.nimp)]
                for imp_i in range(args.nimp):
                    _, Xy_fake_scaled_, _, _, _ = minmax_scale_dummy(Xy_train, data["imp"][method][imp_i], cat_indexes, divide_by=2)
                    Xy_fake_scaled[imp_i] = np.expand_dims(Xy_fake_scaled_, axis=0)
                    Xy_fake[imp_i] = np.expand_dims(data["imp"][method][imp_i], axis=0)
                Xy_fake_scaled = np.concatenate(Xy_fake_scaled, axis=0) # [nimp, n, p]
                Xy_fake = np.concatenate(Xy_fake, axis=0) # [nimp, n, p]

                # concatenate the fakes only at missing values
                Xy_fake_scaled_obs = []
                for imp_i in range(args.nimp):
                    Xy_fake_scaled_obs.append(np.expand_dims(Xy_fake_scaled[imp_i][mask_scaled.astype(bool)], axis=0))
                Xy_fake_scaled_obs = np.concatenate(Xy_fake_scaled_obs, axis=0) # [nimp, nmiss]

                # Get median, mode accross imputations
                median_mode_fake = np.zeros(Xy_train.shape)
                median_mode_fake[:, not_cat_indexes] = np.median(Xy_fake[:, :, not_cat_indexes], axis=0, keepdims=False) # [nimp, n, p_continuous] - > [n, p_continuous]
                median_mode_fake[:, cat_indexes] = stats.mode(Xy_fake[:, :, cat_indexes], keepdims=False)[0] # [nimp, n, p_categorical] - > [n, p_categorical]
                # add one-hot categories
                _, median_mode_scaled, _, _, _ = minmax_scale_dummy(Xy_train, median_mode_fake, cat_indexes, divide_by=2)

                # Mean-variance accross different imputations
                mean_var[method] += np.mean(np.var(Xy_fake_scaled_obs, axis=0)) / args.nexp
                # Mean absolute deviation around the mean, makes more sense given the Gower distance (equivalent to Gower distance between data and the median-mode for each variable)
                mean_mad_mean[method] += np.sum(np.absolute(Xy_fake_scaled - np.mean(Xy_fake_scaled, axis=0, keepdims=False))) / (n_miss*args.nexp)
                # Mean absolute deviation around the median/mode, makes more sense given the Gower distance (equivalent to Gower distance between data and the median-mode for each variable)
                mean_mad_median[method] += np.sum(np.absolute(Xy_fake_scaled - median_mode_scaled)) / (n_miss*args.nexp)


                # Mean absolute error to the ground truth (note: we divide by the n_miss from before the one-hot-encoding)
                # Minimum(MAE): favorizes uncertainty-based methods
                score_mae_min[method] += MAE_min(Xy_train_scaled, Xy_fake_scaled, mask_scaled, n_miss=n_miss) / args.nexp
                # Average(MEA): favorizes single-imputation methods
                for imp_i in range(args.nimp):
                    score_mae_avg[method] += MAE(Xy_train_scaled, Xy_fake_scaled[imp_i], mask_scaled, n_miss=n_miss) / (args.nexp*args.nimp)

                # Statistical measures
                X_fake = []
                y_fake = []
                for imp_i in range(args.nimp):
                    X_fake.append(np.expand_dims(data["imp"][method][imp_i][:,:-1], axis=0))
                    y_fake.append(np.expand_dims(data["imp"][method][imp_i][:,-1], axis=0))
                X_fake = np.concatenate(X_fake, axis=0) # [nimp, n, p-1]
                y_fake = np.concatenate(y_fake, axis=0) # [nimp, n, 1]
                if not cat_y and not bin_y: # too unstable with class and multiclass to due quasi-seperation
                    percent_bias_, coverage_rate_, AW_ = test_imputation_regression(X_train, y_train, X_fake, y_fake, cat_indexes=cat_indexes_no_y, type_model='regression')
                else: 
                    percent_bias_, coverage_rate_, AW_ = 0.0, 0.0, 0.0
                percent_bias[method] += percent_bias_ / args.nexp
                coverage_rate[method] += coverage_rate_ / args.nexp
                AW[method] += AW_ / args.nexp

                for imp_i in range(args.nimp):

                    # Wasserstein-2 Distance based on L1 cost (after scaling)
                    if Xy_train.shape[0] < OTLIM:
                        score_W1_miss[method] += pot.emd2(pot.unif(Xy_train_scaled[M_scaled].shape[0]), pot.unif(Xy_fake_scaled[imp_i][M_scaled].shape[0]), M = pot.dist(Xy_train_scaled[M_scaled], Xy_fake_scaled[imp_i][M_scaled], metric='cityblock')) / (args.nexp*args.nimp)
                        score_W1_train[method] += pot.emd2(pot.unif(Xy_train_scaled.shape[0]), pot.unif(Xy_fake_scaled[imp_i].shape[0]), M = pot.dist(Xy_train_scaled, Xy_fake_scaled[imp_i], metric='cityblock')) / (args.nexp*args.nimp)
                        score_W1_test[method] += pot.emd2(pot.unif(Xy_test_scaled.shape[0]), pot.unif(Xy_fake_scaled[imp_i].shape[0]), M = pot.dist(Xy_test_scaled, Xy_fake_scaled[imp_i], metric='cityblock')) / (args.nexp*args.nimp)

                    # Trained on imputed data
                    X_fake, y_fake = data["imp"][method][imp_i][:,:-1], data["imp"][method][imp_i][:,-1]
                    f1_imp, R2_imp = test_on_multiple_models(X_fake, y_fake, X_test, y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)

                    for key in['mean', 'lin', 'linboost', 'tree', 'treeboost']:
                        f1[method][key] += f1_imp[key] / (args.nexp*args.nimp)
                        R2[method][key] += R2_imp[key] / (args.nexp*args.nimp)

            # Write results in csv file
            # Columns: dataset , method , score_rmse , score_W1 , mean_var , R2_oracle, R2_imp , f1_oracle, f1_imp
            mask_str = f"MCAR({p}) "
            if method in ['forest_diffusion', 'forest_knn', 'forest_diffusion_repaint']:
                method_str = f"{method} n_t={args.n_t} n_t_sampling={args.n_t_sampling} model={args.forest_model} diffusion={args.diffusion_type} duplicate_K={args.duplicate_K} ycond={args.ycond} "
                if args.forest_model == 'xgboost':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} eta={args.eta} "
                elif args.forest_model == 'random_forest':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} "
                elif args.forest_model == 'catboost':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} "
                elif args.forest_model == 'lgbm':
                    method_str += f"num_leaves={args.num_leaves} n_trees={args.n_estimators} lr={args.eta} "
            elif method in ['KNN', 'KNN_std', 'KNN_scaled', 'KNN_scaled2']:
                method_str = f"{method}(n_neighbors={args.n_neighbors}) "
            else:
                method_str = f"{method} "
            csv_str = f"{dataset} , " + f"{mask_str}, " + method_str + f", {score_mae_min[method]} , {score_mae_avg[method]} , {percent_bias[method]} , {coverage_rate[method]} , {score_W1_train[method]} , {score_W1_test[method]} , {mean_var[method]} , {mean_mad_mean[method]} , {mean_mad_median[method]} , {R2[method]['mean']} , {f1[method]['mean']} , {time_taken[method]} "
            for key in['lin', 'linboost', 'tree', 'treeboost']:
                csv_str += f", {R2[method][key]} , {f1[method][key]} "
            csv_str += f"\n"

            print(csv_str)
            with open(args.out_path, 'a+') as f: # where we keep track of the results
                f.write(csv_str)

        method_index_start = 0 #  so we loop back again