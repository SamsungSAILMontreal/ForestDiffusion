import tomli
import shutil
import os
import copy
import argparse
from TabDDPM.scripts.train import train
from TabDDPM.scripts.sample import sample
import pandas as pd
import matplotlib.pyplot as plt
import TabDDPM.lib as lib
import torch
import numpy as np
import shutil
from pathlib import Path
import gc
import time

def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main_fn(config='TabDDPM/exp/my_data/config.toml', 
    cat_indexes=[], d_in=0, num_classes=0, num_samples=0, num_numerical_features=0, seed=0, ngen=1):

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = {}
    args['config']=config
    args = dotdict(args)

    raw_config = lib.load_config(args.config)
    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda:0')

    raw_config['num_numerical_features'] = num_numerical_features
    raw_config['model_params']['num_classes'] = num_classes
    raw_config['model_params']['is_y_cond'] = num_classes > 0 # only for classification
    raw_config['sample']['num_samples'] = num_samples

    dataset_dir = Path(raw_config['parent_dir'])
    if not dataset_dir.exists():
        dataset_dir.mkdir()
    else: # we delete before starting
        shutil.rmtree(dataset_dir)
        dataset_dir.mkdir()

    start = time.time()

    train(
        **raw_config['train']['main'],
        **raw_config['diffusion_params'],
        parent_dir=raw_config['parent_dir'],
        real_data_path=raw_config['real_data_path'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        seed=seed,
        change_val=False
    )
    sample(
        num_samples=raw_config['sample']['num_samples']*ngen,
        batch_size=raw_config['sample']['batch_size'],
        disbalance=raw_config['sample'].get('disbalance', None),
        **raw_config['diffusion_params'],
        parent_dir=raw_config['parent_dir'],
        real_data_path=raw_config['real_data_path'],
        model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        seed=seed,
        change_val=False
    )
    torch.cuda.empty_cache()
    gc.collect()

    y = np.load('TabDDPM/exp/my_data/y_train.npy', allow_pickle=True)

    try: # There exist continuous variables
        X_num = np.load('TabDDPM/exp/my_data/X_num_train.npy', allow_pickle=True)
        X_comb = copy.deepcopy(X_num)
        try: # Add categorical variables if they exist
            X_cat = np.load('TabDDPM/exp/my_data/X_cat_train.npy', allow_pickle=True)
            for j, i in enumerate(cat_indexes):
                if j < X_cat.shape[1]: # 0, 1, 2 for len=3
                    X_comb = np.insert(X_comb, i, X_cat[:,j], axis=1)
        except:
            pass
    except:
        # all variables are categorical
        X_comb = np.load('TabDDPM/exp/my_data/X_cat_train.npy', allow_pickle=True)

    X_comb = np.concatenate((X_comb, np.expand_dims(y, axis=1)), axis=1)

    print(f'Elapsed time: {time.time() - start}')

    return X_comb
