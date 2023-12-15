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
from data_loaders import dataset_loader
from sklearn.model_selection import train_test_split
import argparse
from ForestDiffusion import ForestDiffusionModel
from metrics import test_on_multiple_models, compute_coverage, test_imputation_regression, test_on_multiple_models_classifier
from STaSy.stasy import STaSy_model
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from CTABGANPlus.ctabgan import CTABGAN
from TabDDPM.scripts.pipeline import main_fn as tab_ddpm_fn
from TabDDPM.lib.dataset_prep import my_data_prep
import miceforest as mf
from missforest import MissForest

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
parser.add_argument('--out_path', type=str, default='jolicoea/tabular_generation_results.txt',
                    help='filename for the results')

parser.add_argument("--restore_from_name", type=str2bool, default=False, help="if True, restore session based on name")
parser.add_argument("--name", type=str, default='my_exp', help="used when restoring from crashed instances")

parser.add_argument("--methods", type=str, nargs='+', default=['oracle', 'CTGAN', 'GaussianCopula', 'TVAE', 'CopulaGAN', 'CTABGAN', 'stasy', 'TabDDPM', 'forest_diffusion'], help="oracle, CTGAN, GaussianCopula, TVAE, CopulaGAN, CTABGAN, stasy, TabDDPM, forest_diffusion")
parser.add_argument('--nexp', type=int, default=3,
                    help='number of experiences per parameter setting')
parser.add_argument('--ngen', type=int, default=5,
                    help='number of generations per method')
parser.add_argument('--n_tries', type=int, default=5,
                    help='number of models trained with different seeds in the metrics')
parser.add_argument('--datasets', nargs='+', type=str, default=['iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'california', 'bean', 'tictactoe','congress','car'],
                    help='datasets on which to run the experiments')

# Setting for Missingness if used
parser.add_argument('--add_missing_data', type=str2bool, default=False)
parser.add_argument('--p', type=float, default=0.2, help='Proportion of missing')
parser.add_argument('--imputation_method', type=str, default='MissForest', help='miceforest or MissForest or none (MissForest is better and the one used in the paper for the non-ForestDiffusion methods; ForestDiffusion is the only method that can handle none)')

# Forest hyperparameters
parser.add_argument('--forest_model', type=str, default='xgboost', help='xgboost, random_forest, lgbm, catboost')
parser.add_argument('--diffusion_type', type=str, default='vp', help='flow (flow-matching), vp (Variance-Preserving diffusion)')
parser.add_argument('--n_t', type=int, default=50, help='number of times t in [0,1]')
parser.add_argument('--n_t_sampling', type=int, default=0, help='number of times t in [0,1] for sampling  (0 will uses n_t steps; ignore this parameter honestly, its worth changing)')
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
parser.add_argument('--n_jobs', type=int, default=-1, help='')
parser.add_argument('--n_batch', type=int, default=1, help='If >0 use the data iterator with the specified number of batches')

# stasy hyperparameters
parser.add_argument('--act', type=str, default='elu', help='')
parser.add_argument('--layer_type', type=str, default='concatsquash', help='')
parser.add_argument('--sde', type=str, default='vesde', help='')
parser.add_argument('--lr', type=float, default=2e-3, help='')
parser.add_argument('--num_scales', type=int, default=50, help='')

args = parser.parse_args()


if __name__ == "__main__":

    if args.n_t_sampling == 0:
        args.n_t_sampling = args.n_t

    OTLIM = 5000

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
        if dataset == 'ecoli' and args.add_missing_data:
            print("Ecoli with missing data can causes problems in the metrics due to near-constant variables, skipping it")
            continue
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

        # Integers
        int_indexes = []
        if int_x is not None:
            int_indexes = int_indexes + int_x
        int_indexes_no_y = copy.deepcopy(int_indexes)
        if int_y:
            int_indexes.append(X.shape[1])

        score_W1_train = {}
        score_W1_test = {}
        coverage = {}
        coverage_test = {}
        time_taken = {}
        percent_bias = {}
        coverage_rate = {}
        AW = {}
        f1_class = {}
        for method in args.methods:
            score_W1_test[method] = 0.0
            score_W1_train[method] = 0.0
            coverage[method] = 0.0
            coverage_test[method] = 0.0
            time_taken[method] = 0.0
            percent_bias[method] = 0.0
            coverage_rate[method] = 0.0
            AW[method] = 0.0
            f1_class[method] = []

        R2 = {}
        f1 = {}
        for method in args.methods:
            R2[method] = {'real': {}, 'fake': {}, 'both': {}}
            f1[method] = {'real': {}, 'fake': {}, 'both': {}}
            for test_type in ['real','fake','both']:
                for test_type2 in ['mean','lin','linboost', 'tree', 'treeboost']:
                    R2[method][test_type][test_type2] = 0.0
                    f1[method][test_type][test_type2] = 0.0

        for method_index in range(method_index_start, len(args.methods)):
            
            method = args.methods[method_index]
            print(f'method={method}')

            with open(args.name, 'w') as f: # where we track where we are, to restore sessions after crashes
                f.write(f'{dataset_index}&{method_index}')

            for n in range(args.nexp):

                print(n)

                # Need to train/test split for evaluating the linear regression performance and for W1 based on test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n, stratify=y if bin_y or cat_y else None)
                Xy_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
                Xy_test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)

                if args.add_missing_data:
                    print("Adding missing data")

                    torch.manual_seed(n)
                    np.random.seed(n)

                    if torch.cuda.is_available():
                        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
                    else:
                        torch.set_default_tensor_type('torch.DoubleTensor')

                    ### Each entry from the second axis has a probability p of being NA 
                    X_true = torch.tensor(X_train)
                    mask_x = (torch.rand(X_true.shape) < args.p).double()

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

                    # We must impute the data first
                    if args.imputation_method == 'MissForest':
                        data_nas_ = copy.deepcopy(data_nas)
                        imputer = MissForest(random_state=0)
                        Xy_train_used = imputer.fit_transform(data_nas_, cat_vars=bin_indexes + cat_indexes if len(bin_indexes + cat_indexes) > 0 else None)
                    elif args.imputation_method == 'miceforest':
                        # Convert to Pandas
                        data_pd = pd.DataFrame(data_nas, columns = [str(i) for i in range(data_nas.shape[1])])
                        # indicate which column is categorical so that they are handled properly
                        for column_k in bin_indexes + cat_indexes:
                            data_pd[str(column_k)] = data_pd[str(column_k)].astype('category') 
                        kds = mf.ImputationKernel(data_pd, save_all_iterations=False, datasets=1, random_state=n)
                        kds.mice(5) # 5 iterations is the default and should be enough
                        Xy_train_used = kds.complete_data(dataset=0).to_numpy()
                    elif args.imputation_method == "none":
                        assert method == "forest_diffusion"
                        Xy_train_used = data_nas
                    else:
                        raise NotImplementedError("imputation_method must be MissForest or miceforest")

                else: # no missing data
                    Xy_train_used = Xy_train

                if method in ['TabDDPM','TVAE'] or not torch.cuda.is_available():
                    torch.set_default_tensor_type('torch.FloatTensor')
                else:
                    torch.set_default_tensor_type('torch.cuda.FloatTensor')

                start = time.time()

                if method == 'oracle':
                    Xy_fake = np.tile(np.expand_dims(Xy_train, axis=0), reps=(args.ngen, 1, 1)) # [ngen, n, p]

                elif method == 'CTGAN':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = CTGANSynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'GaussianCopula':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = GaussianCopulaSynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'TVAE':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = TVAESynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'CopulaGAN':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = CopulaGANSynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'CTABGAN': # CTABGAN+

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    synthesizer =  CTABGAN(pd_data = data_pd,
                                     categorical_columns = [str(i) for i in cat_indexes + bin_indexes],  
                                     general_columns= [str(i) for i in range(Xy_train_used.shape[1]) if i not in cat_indexes + bin_indexes + int_indexes],
                                     integer_columns = [str(i) for i in int_indexes]) 
                    synthesizer.fit()

                    def my_synthesizer():
                        synthetic_data = synthesizer.generate_samples()
                        return synthetic_data.to_numpy().astype('float')
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'stasy':
                    Xy_fake = STaSy_model(Xy_train_used,
                       categorical_columns=cat_indexes + bin_indexes, 
                       ordinal_columns=int_indexes, 
                       seed=n, 
                       epochs=10000,
                       ngen=args.ngen,
                       activation = args.act, layer_type = args.layer_type, sde = args.sde, lr = args.lr, num_scales = args.num_scales)
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]

                elif method == 'TabDDPM': # TabDDPM

                    # Prep the data, will be save in the format that TabDDPM wants
                    columns = [str(i) for i in range(Xy_train_used.shape[1])]
                    data_pd = pd.DataFrame(Xy_train_used, columns = columns)
                    X_pd = data_pd[columns[0:-1]]
                    y_pd = data_pd[columns[-1]]
                    cat_ind = [str(i) for i in range(X_pd.shape[1]) if i in cat_indexes+bin_indexes]
                    noncat_ind = [str(i) for i in range(X_pd.shape[1]) if i not in cat_indexes+bin_indexes]
                    if cat_y:
                        task_type='multiclass'
                    elif bin_y:
                        task_type='binclass'
                    else:
                        task_type='regression'
                    my_data_prep(X_pd, y_pd, task=task_type, cat_ind=cat_ind, noncat_ind=noncat_ind)
                    synthetic_data = tab_ddpm_fn(config='TabDDPM/config/config.toml', 
                        cat_indexes=cat_indexes+bin_indexes, num_classes=len(np.unique(y_pd)) if cat_y or bin_y else 0, 
                        num_samples=Xy_train_used.shape[0], num_numerical_features=len(noncat_ind), seed=n, ngen=args.ngen)
                    Xy_fake = synthetic_data.astype('float')
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]

                elif method == 'forest_diffusion':

                    if args.ycond and (bin_y or cat_y):
                        forest_model = ForestDiffusionModel(X=Xy_train_used[:,:-1], 
                            label_y=Xy_train_used[:,-1],
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
                        forest_model = ForestDiffusionModel(X=Xy_train_used, 
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
                    Xy_fake = forest_model.generate(batch_size=args.ngen*Xy_train_used.shape[0], n_t=args.n_t_sampling)
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]
                end = time.time()
                time_taken[method] += (end - start) / args.nexp
                assert Xy_fake.shape[0] == args.ngen and Xy_fake.shape[1] == Xy_train_used.shape[0] and Xy_fake.shape[2] == Xy_train_used.shape[1]

                for gen_i in range(args.ngen):

                    #np.set_printoptions(threshold=np.inf)
                    #print(Xy_train[0:150])
                    #print(Xy_fake[gen_i][0:150])

                    Xy_fake_i = Xy_fake[gen_i]

                    # Mixed data is tricky, nearest neighboors (for the coverage) and Wasserstein distance (based on L2) are not scale invariant
                    # To ensure that the scaling between variables is relatively uniformized, we take inspiration from the Gower distance used in mixed-data KNNs: https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b
                    # Continuous: we do min-max normalization (to use Gower |x1-x2|/(max-min) as distance)
                    # Categorical: We one-hot and then divide by 2 (e.g., 0 0 0.5 with 0.5 0 0 will have distance 0.5 + 0.5 = 1)
                    # After these transformations, taking the L1 (City-block / Manhattan distance) norm distance will give the Gower distance
                    Xy_train_scaled, Xy_fake_scaled, _, _, _ = minmax_scale_dummy(Xy_train, Xy_fake_i, cat_indexes, divide_by=2)
                    _, Xy_test_scaled, _, _, _ = minmax_scale_dummy(Xy_train, Xy_test, cat_indexes, divide_by=2)

                    # Wasserstein-1 based on L1 cost (after scaling)
                    if Xy_train.shape[0] < OTLIM:
                        score_W1_train[method] += pot.emd2(pot.unif(Xy_train_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_train_scaled, Xy_fake_scaled, metric='cityblock')) / (args.nexp*args.ngen)
                        score_W1_test[method] += pot.emd2(pot.unif(Xy_test_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_test_scaled, Xy_fake_scaled, metric='cityblock')) / (args.nexp*args.ngen)

                    X_fake, y_fake = Xy_fake_i[:,:-1], Xy_fake_i[:,-1]

                    # Trained on real data
                    f1_real, R2_real = test_on_multiple_models(X_train, y_train, X_test, y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)

                    # Trained on fake data
                    f1_fake, R2_fake = test_on_multiple_models(X_fake, y_fake, X_test, y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)

                    # Trained on real data and fake data
                    X_both = np.concatenate((X_train,X_fake), axis=0)
                    y_both = np.concatenate((y_train,y_fake))
                    f1_both, R2_both = test_on_multiple_models(X_both, y_both, X_test, y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)
                    
                    for key in['mean', 'lin', 'linboost', 'tree', 'treeboost']:
                        f1[method]['real'][key] += f1_real[key] / (args.nexp*args.ngen)
                        f1[method]['fake'][key] += f1_fake[key] / (args.nexp*args.ngen)
                        f1[method]['both'][key] += f1_both[key] / (args.nexp*args.ngen)
                        R2[method]['real'][key] += R2_real[key] / (args.nexp*args.ngen)
                        R2[method]['fake'][key] += R2_fake[key] / (args.nexp*args.ngen)
                        R2[method]['both'][key] += R2_both[key] / (args.nexp*args.ngen)

                    # Get another different fake data for use as test fake-data
                    Xy_fake_j = Xy_fake[(gen_i + 1) % args.ngen] # 0 -> 1, 1-> 2, n -> 0
                    # Classifier comparing real to fake data, the less it classify fake data as fake = the better
                    f1_class[method] += [test_on_multiple_models_classifier(X_train_real=Xy_train, X_train_fake=Xy_fake_i, X_test_fake=Xy_fake_j, cat_indexes=cat_indexes, nexp=args.n_tries)]

                    # coverage based on L1 cost (after scaling)
                    coverage[method] += compute_coverage(Xy_train_scaled, Xy_fake_scaled, None) / (args.nexp*args.ngen)
                    coverage_test[method] += compute_coverage(Xy_test_scaled, Xy_fake_scaled, None) / (args.nexp*args.ngen)

                # Statistical measures
                X_fake = []
                y_fake = []
                for gen_i in range(args.ngen):
                    X_fake.append(np.expand_dims(Xy_fake[gen_i][:,:-1], axis=0))
                    y_fake.append(np.expand_dims(Xy_fake[gen_i][:,-1], axis=0))
                X_fake = np.concatenate(X_fake, axis=0) # [nimp, n, p-1]
                y_fake = np.concatenate(y_fake, axis=0) # [nimp, n, 1]
                # Too unstable with classification due toquasi-seperation with logistic regression
                # dataset=ecoli with missing data is removed because it has near-constant variables, and the non-constant parts can be lost when adding missing data making it perfectly multicorrelated which will give regression errors
                if not cat_y and not bin_y and not (dataset == 'ecoli' and args.add_missing_data):
                    percent_bias_, coverage_rate_, AW_ = test_imputation_regression(X_train, y_train, X_fake, y_fake, 
                        cat_indexes=cat_indexes_no_y, type_model='regression')
                else: 
                    percent_bias_, coverage_rate_, AW_ = 0.0, 0.0, 0.0
                percent_bias[method] += percent_bias_ / args.nexp
                coverage_rate[method] += coverage_rate_ / args.nexp
                AW[method] += AW_ / args.nexp

            # Write results in csv file
            # Columns: dataset , method , score_W1_train , score_W1_test , R2_real , R2_fake , f1_real , f1_fake, coverage
            if method == 'forest_diffusion':
                method_str = f"{method} n_t={args.n_t} n_t_sampling={args.n_t_sampling} model={args.forest_model} diffusion={args.diffusion_type} duplicate_K={args.duplicate_K} ycond={args.ycond} "
                if args.forest_model == 'xgboost':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} eta={args.eta} "
                elif args.forest_model == 'random_forest':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} "
                elif args.forest_model == 'catboost':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} "
                elif args.forest_model == 'lgbm':
                    method_str += f"num_leaves={args.num_leaves} n_trees={args.n_estimators} lr={args.eta} "
            else:
                method_str = f"{method} "
            if args.add_missing_data:
                mask_str = f"MCAR({args.p} {args.imputation_method}) "
                csv_str = f"{dataset} , " + f"{mask_str}, " + method_str + f", {score_W1_train[method]} , {score_W1_test[method]} , {R2[method]['real']['mean']} , {R2[method]['fake']['mean']} , {R2[method]['both']['mean']} , {f1[method]['real']['mean']} , {f1[method]['fake']['mean']} , {f1[method]['both']['mean']} , {coverage[method]} , {coverage_test[method]} , {percent_bias[method]} , {coverage_rate[method]} , {AW[method]} , {f1_class[method]}  , {time_taken[method]} "
            else:
                csv_str = f"{dataset} , " + method_str + f", {score_W1_train[method]} , {score_W1_test[method]} , {R2[method]['real']['mean']} , {R2[method]['fake']['mean']} , {R2[method]['both']['mean']} , {f1[method]['real']['mean']} , {f1[method]['fake']['mean']} , {f1[method]['both']['mean']} , {coverage[method]} , {coverage_test[method]} , {percent_bias[method]} , {coverage_rate[method]} , {AW[method]} , {f1_class[method]}  , {time_taken[method]} "
            for key in['lin', 'linboost', 'tree', 'treeboost']:
                csv_str += f", {R2[method]['real'][key]} , {R2[method]['fake'][key]} , {R2[method]['both'][key]} , {f1[method]['real'][key]} , {f1[method]['fake'][key]} , {f1[method]['both'][key]} "
            csv_str += f"\n"
            print(csv_str)
            with open(args.out_path, 'a+') as f: # where we keep track of the results
                f.write(csv_str)
        method_index_start = 0 #  so we loop back again