#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris, load_wine, fetch_california_housing
import zipfile
import os
import pandas as pd
import numpy as np
import wget


DATASETS = ['iris', 'wine', 'california', 'parkinsons', \
            'climate_model_crashes', 'concrete_compression', \
            'yacht_hydrodynamics', 'airfoil_self_noise', \
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
            'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', \
            'blood_transfusion', 'breast_cancer_diagnostic', \
            'connectionist_bench_vowel', 'concrete_slump', \
            'wine_quality_red', 'wine_quality_white', \
            'bean', 'tictactoe','congress','car', 'higgs']

def dataset_loader(dataset):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes 
    datasets are located in './datasets'. If the called for dataset is not in 
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.
        
    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """
    assert dataset in DATASETS , f"Dataset not supported: {dataset}"

    if not os.path.isdir('datasets'):
        os.mkdir('datasets')

    if dataset in DATASETS:
        bin_y = False # binary outcome
        cat_y = False # categorical w/ >=2 outcome
        int_y = False # integer outcome

        bin_x = None # binary
        cat_x = None # categorical w/ >=2 classes
        int_x = None # integers

        if dataset == 'iris':
            my_data = load_iris()
            cat_y = True
        elif dataset == 'wine':
            my_data = load_wine()
            cat_y = True
        #elif dataset == 'boston': # not part of sklearn anymore
        #    my_data = load_boston()
        elif dataset == 'california':
            my_data = fetch_california_housing()
            int_x = [1, 4]
        elif dataset == 'parkinsons':
            my_data = fetch_parkinsons()
            bin_y = True
        elif dataset == 'climate_model_crashes':
            my_data = fetch_climate_model_crashes()
            bin_y = True
        elif dataset == 'concrete_compression':
            my_data = fetch_concrete_compression()
            int_x = [7]
        elif dataset == 'yacht_hydrodynamics':
            my_data = fetch_yacht_hydrodynamics()
        elif dataset == 'airfoil_self_noise':
            my_data = fetch_airfoil_self_noise()
        elif dataset == 'connectionist_bench_sonar':
            my_data = fetch_connectionist_bench_sonar()
            bin_y = True
        elif dataset == 'ionosphere':
            my_data = fetch_ionosphere()
            bin_x = [0]
            bin_y = True
        elif dataset == 'qsar_biodegradation':
            my_data = fetch_qsar_biodegradation()
            int_x = [2,3,4,5,6,8,9,10,15,18,19,20,22,25,31,32,33,34,37,39,40]
            bin_x = [23,24,28]
            bin_y = True
        elif dataset == 'seeds':
            my_data = fetch_seeds()
            cat_y = True
        elif dataset == 'glass':
            my_data = fetch_glass()
            cat_y = True
        elif dataset == 'ecoli':
            my_data = fetch_ecoli()
            cat_y = True
        elif dataset == 'yeast':
            my_data = fetch_yeast()
            cat_y = True
        elif dataset == 'libras':
            my_data = fetch_libras()
            cat_y = True
        elif dataset == 'planning_relax':
            my_data = fetch_planning_relax()
            bin_y = True
        elif dataset == 'blood_transfusion':
            my_data = fetch_blood_transfusion()
            int_x = [0,1,3]
            bin_y = True
        elif dataset == 'breast_cancer_diagnostic':
            my_data = fetch_breast_cancer_diagnostic()
            bin_y = True
        elif dataset == 'connectionist_bench_vowel':
            my_data = fetch_connectionist_bench_vowel()
            bin_y = True
        elif dataset == 'concrete_slump':
            my_data = fetch_concrete_slump()
        elif dataset == 'wine_quality_red':
            int_y = True
            my_data = fetch_wine_quality_red()
        elif dataset == 'wine_quality_white':
            int_y = True
            my_data = fetch_wine_quality_white()
        elif dataset == 'bean':
            my_data = fetch_bean()
            int_x = [0,6]
            cat_y = True
        elif dataset == 'tictactoe': # all categorical
            my_data = fetch_tictactoe()
            cat_x = [0,1,2,3,4,5,6,7,8]
            bin_y = True
        elif dataset == 'congress': # all categorical
            my_data = fetch_congress()
            cat_x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            cat_y = True
        elif dataset == 'car': # all categorical
            my_data = fetch_car()
            cat_x = [0,1,2,3,4,5]
            cat_y = True
        else:
            raise Exception('dataset does not exists')
        X, y = my_data['data'], my_data['target']

        return X, bin_x, cat_x, int_x, y, bin_y, cat_y, int_y

def fetch_parkinsons():
    if not os.path.isdir('datasets/parkinsons'):
        os.mkdir('datasets/parkinsons')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        wget.download(url, out='datasets/parkinsons/')

    with open('datasets/parkinsons/parkinsons.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = 0)
        Xy = {}
        Xy['data'] = np.concatenate((df.values[:, 1:17].astype('float'), df.values[:, 18:].astype('float')), axis=1)
        Xy['target'] =  pd.factorize(df.values[:, 17])[0] # str to numeric

    return Xy


def fetch_climate_model_crashes():
    if not os.path.isdir('datasets/climate_model_crashes'):
        os.mkdir('datasets/climate_model_crashes')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat'
        wget.download(url, out='datasets/climate_model_crashes/')

    with open('datasets/climate_model_crashes/pop_failures.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, 2:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_concrete_compression():
    if not os.path.isdir('datasets/concrete_compression'):
        os.mkdir('datasets/concrete_compression')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        wget.download(url, out='datasets/concrete_compression/')

    with open('datasets/concrete_compression/Concrete_Data.xls', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy


def fetch_yacht_hydrodynamics():
    if not os.path.isdir('datasets/yacht_hydrodynamics'):
        os.mkdir('datasets/yacht_hydrodynamics')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
        wget.download(url, out='datasets/yacht_hydrodynamics/')

    with open('datasets/yacht_hydrodynamics/yacht_hydrodynamics.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy

def fetch_airfoil_self_noise():
    if not os.path.isdir('datasets/airfoil_self_noise'):
        os.mkdir('datasets/airfoil_self_noise')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        wget.download(url, out='datasets/airfoil_self_noise/')

    with open('datasets/airfoil_self_noise/airfoil_self_noise.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy


def fetch_connectionist_bench_sonar():
    if not os.path.isdir('datasets/connectionist_bench_sonar'):
        os.mkdir('datasets/connectionist_bench_sonar')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        wget.download(url, out='datasets/connectionist_bench_sonar/')

    with open('datasets/connectionist_bench_sonar/sonar.all-data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_ionosphere():
    if not os.path.isdir('datasets/ionosphere'):
        os.mkdir('datasets/ionosphere')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out='datasets/ionosphere/')

    with open('datasets/ionosphere/ionosphere.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = np.concatenate((df.values[:, 0:1].astype('float'), df.values[:, 2:-1].astype('float')), axis=1) # removing the secon variable which is always 0
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_qsar_biodegradation():
    if not os.path.isdir('datasets/qsar_biodegradation'):
        os.mkdir('datasets/qsar_biodegradation')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out='datasets/qsar_biodegradation/')

    with open('datasets/qsar_biodegradation/biodeg.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_seeds():
    if not os.path.isdir('datasets/seeds'):
        os.mkdir('datasets/seeds')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
        wget.download(url, out='datasets/seeds/')

    with open('datasets/seeds/seeds_dataset.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1] - 1 # make 0, 1, 2 instead of 1, 2, 3

    return Xy


def fetch_glass():
    if not os.path.isdir('datasets/glass'):
        os.mkdir('datasets/glass')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        wget.download(url, out='datasets/glass/')

    with open('datasets/glass/glass.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  (df.values[:, -1] - 1).astype('int') # make 0, 1, 2 instead of 1, 2, 3
        Xy['target'][Xy['target'] >= 4] = Xy['target'][Xy['target'] >= 4] - 1 # 0, 1, 2, 4, 5, 6 -> 0, 1, 2, 3, 4, 5

    return Xy


def fetch_ecoli():
    if not os.path.isdir('datasets/ecoli'):
        os.mkdir('datasets/ecoli')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
        wget.download(url, out='datasets/ecoli/')

    with open('datasets/ecoli/ecoli.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy

def fetch_yeast():
    if not os.path.isdir('datasets/yeast'):
        os.mkdir('datasets/yeast')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        wget.download(url, out='datasets/yeast/')

    with open('datasets/yeast/yeast.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy


def fetch_libras():
    if not os.path.isdir('datasets/libras'):
        os.mkdir('datasets/libras')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
        wget.download(url, out='datasets/libras/')

    with open('datasets/libras/movement_libras.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1] - 1  # make 0, 1, 2 instead of 1, 2, 3

    return Xy

def fetch_planning_relax():
    if not os.path.isdir('datasets/planning_relax'):
        os.mkdir('datasets/planning_relax')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt'
        wget.download(url, out='datasets/planning_relax/')

    with open('datasets/planning_relax/plrx.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1] - 1

    return Xy


def fetch_blood_transfusion():
    if not os.path.isdir('datasets/blood_transfusion'):
        os.mkdir('datasets/blood_transfusion')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        wget.download(url, out='datasets/blood_transfusion/')

    with open('datasets/blood_transfusion/transfusion.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_breast_cancer_diagnostic():
    if not os.path.isdir('datasets/breast_cancer_diagnostic'):
        os.mkdir('datasets/breast_cancer_diagnostic')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out='datasets/breast_cancer_diagnostic/')

    with open('datasets/breast_cancer_diagnostic/wdbc.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] = pd.factorize(df.values[:, 1])[0] # str to numeric

    return Xy


def fetch_connectionist_bench_vowel():
    if not os.path.isdir('datasets/connectionist_bench_vowel'):
        os.mkdir('datasets/connectionist_bench_vowel')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        wget.download(url, out='datasets/connectionist_bench_vowel/')

    with open('datasets/connectionist_bench_vowel/vowel-context.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 3:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_concrete_slump():
    if not os.path.isdir('datasets/concrete_slump'):
        os.mkdir('datasets/concrete_slump')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
        wget.download(url, out='datasets/concrete_slump/')

    with open('datasets/concrete_slump/slump_test.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, 1:-3].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float') # the 3 last variables are actually outcomes, but we choose 1, because we can't have 3!

    return Xy


def fetch_wine_quality_red():
    if not os.path.isdir('datasets/wine_quality_red'):
        os.mkdir('datasets/wine_quality_red')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        wget.download(url, out='datasets/wine_quality_red/')

    with open('datasets/wine_quality_red/winequality-red.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy


def fetch_wine_quality_white():
    if not os.path.isdir('datasets/wine_quality_white'):
        os.mkdir('datasets/wine_quality_white')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        wget.download(url, out='datasets/wine_quality_white/')

    with open('datasets/wine_quality_white/winequality-white.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('float')

    return Xy

def fetch_bean():
    if not os.path.isdir('datasets/DryBeanDataset'):
        os.mkdir('datasets/DryBeanDataset')
        url = 'https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip'
        wget.download(url, out='datasets/DryBeanDataset/')

    with zipfile.ZipFile('datasets/DryBeanDataset/dry+bean+dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('datasets')

    with open('datasets/DryBeanDataset/Dry_Bean_Dataset.xlsx', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, -1])[0] # str to numeric

    return Xy

def fetch_tictactoe():
    if not os.path.isdir('datasets/tictactoe'):
        os.mkdir('datasets/tictactoe')
        url = 'https://archive.ics.uci.edu/static/public/101/tic+tac+toe+endgame.zip'
        wget.download(url, out='datasets/tictactoe/')

    with zipfile.ZipFile('datasets/tictactoe/tic+tac+toe+endgame.zip', 'r') as zip_ref:
        zip_ref.extractall('datasets/tictactoe')

    with open('datasets/tictactoe/tic-tac-toe.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] =  np.zeros(df.values[:, :-1].shape)
        for i in range(Xy['data'].shape[1]):
            Xy['data'][:, i] = pd.factorize(df.values[:, i])[0]
        Xy['target'] =  pd.factorize(df.values[:, -1])[0]

    return Xy

def fetch_congress():
    if not os.path.isdir('datasets/congress'):
        os.mkdir('datasets/congress')
        url = 'https://archive.ics.uci.edu/static/public/105/congressional+voting+records.zip'
        wget.download(url, out='datasets/congress/')

    with zipfile.ZipFile('datasets/congress/congressional+voting+records.zip', 'r') as zip_ref:
        zip_ref.extractall('datasets/congress')

    with open('datasets/congress/house-votes-84.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] =  np.zeros(df.values[:, 1:].shape)
        for i in range(Xy['data'].shape[1]):
            Xy['data'][:, i] = pd.factorize(df.values[:, i+1])[0]
        Xy['target'] =  pd.factorize(df.values[:, 0])[0]

    return Xy

def fetch_car():
    if not os.path.isdir('datasets/car'):
        os.mkdir('datasets/car')
        url = 'https://archive.ics.uci.edu/static/public/19/car+evaluation.zip'
        wget.download(url, out='datasets/car/')

    with zipfile.ZipFile('datasets/car/car+evaluation.zip', 'r') as zip_ref:
        zip_ref.extractall('datasets/car')

    with open('datasets/car/car.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] =  np.zeros(df.values[:, :-1].shape)
        for i in range(Xy['data'].shape[1]):
            Xy['data'][:, i] = pd.factorize(df.values[:, i])[0]
        Xy['target'] =  pd.factorize(df.values[:, -1])[0]

    return Xy

def fetch_higgs():
    if not os.path.isdir('datasets/higgs'):
        os.mkdir('datasets/higgs')
        url = 'https://archive.ics.uci.edu/static/public/280/higgs.zip'
        wget.download(url, out='datasets/higgs/')

        with zipfile.ZipFile('datasets/higgs/higgs.zip', 'r') as zip_ref:
            zip_ref.extractall('datasets/higgs')

    with gzip.open('datasets/higgs/HIGGS.csv.gz', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, 1:].astype('float')
        Xy['target'] =  pd.factorize(df.values[:, 0])[0] # str to numeric

    return Xy