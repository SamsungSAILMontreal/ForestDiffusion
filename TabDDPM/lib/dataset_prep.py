# %%
import json
import shutil
from pathlib import Path
from copy import deepcopy
from typing import Any, Optional, cast
import numpy as np
import pandas as pd
from typing import Dict
import enum
from collections import ChainMap

ArrayDict = Dict[str, np.ndarray]
Info = Dict[str, Any]

class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

def _start(dirname: str):
    #print(f'>>> {dirname}')
    dataset_dir = Path(dirname)
    if not dataset_dir.exists():
        dataset_dir.mkdir()
    else: # we delete before starting
        shutil.rmtree(dataset_dir)
        dataset_dir.mkdir()
    return dataset_dir


def _make_split(size: int, stratify: Optional[np.ndarray], n_parts: int):
    all_idx = np.arange(size, dtype=np.int64)
    return cast(ArrayDict, {'train': all_idx, 'val': all_idx, 'test': all_idx})

def _apply_split(data: ArrayDict, split: ArrayDict):
    return {k: {part: v[idx] for part, idx in split.items()} for k, v in data.items()}

def _save(
    dataset_dir: Path,
    name: str,
    task_type: TaskType,
    *,
    X_num: Optional[ArrayDict],
    X_cat: Optional[ArrayDict],
    y: ArrayDict,
    idx: Optional[ArrayDict],
    id_: Optional[str] = None,
    id_suffix: str = '--default',
) -> None:
    if id_ is not None:
        assert id_suffix == '--default'
    assert (
        X_num is not None or X_cat is not None
    ), 'At least one type of features must be presented.'
    if X_num is not None:
        X_num = {k: v.astype(np.float32) for k, v in X_num.items()}
    if X_cat is not None:
        X_cat = {k: v.astype(str) for k, v in X_cat.items()}
    if idx is not None:
        idx = {k: v.astype(np.int64) for k, v in idx.items()}
    y = {
        k: v.astype(np.float32 if task_type == TaskType.REGRESSION else np.int64)
        for k, v in y.items()
    }
    if task_type != TaskType.REGRESSION:
        y_unique = {k: set(v.tolist()) for k, v in y.items()}
        assert y_unique['train'] == set(range(max(y_unique['train']) + 1))
        for x in ['val', 'test']:
            assert y_unique[x] <= y_unique['train']
        del x

    info = {
        'name': name,
        'id': (dataset_dir.name + id_suffix) if id_ is None else id_,
        'task_type': task_type.value,
        'n_num_features': (0 if X_num is None else next(iter(X_num.values())).shape[1]),
        'n_cat_features': (0 if X_cat is None else next(iter(X_cat.values())).shape[1]),
    } 
    info = {**info, **{f'{k}_size': len(v) for k, v in y.items()}}
    if task_type == TaskType.MULTICLASS:
        info['n_classes'] = len(set(y['train']))
    (dataset_dir / 'info.json').write_text(json.dumps(info, indent=4))

    for data_name in ['X_num', 'X_cat', 'y', 'idx']:
        data = locals()[data_name]
        if data is not None:
            for k, v in data.items():
                np.save(dataset_dir / f'{data_name}_{k}.npy', v)
    (dataset_dir / 'READY').touch()
    #print('Done\n')

def my_data_prep(X, y, cat_ind, noncat_ind, task='regression'): # 'regression', 'binclass', 'multiclass'
    dataset_dir = _start('my_data')

    df_trainval = X
    df_trainval = cast(pd.DataFrame, df_trainval)
    idx = _make_split(len(df_trainval), y, 2)

    if task == 'binclass':
        task_type = TaskType.BINCLASS
    elif task == 'multiclass':
        task_type = TaskType.MULTICLASS
    elif task == 'regression':
        task_type = TaskType.REGRESSION
    else:
        raise Exception("TaskType does not exists")

    X_cat = df_trainval.loc[:, cat_ind].values
    X_num = df_trainval.loc[:, noncat_ind].values
    if X_cat.shape[1] == 0: # no categorical
        X_cat = None
        _save(
            dataset_dir,
            'my_data',
            task_type,
            **_apply_split({'X_num': X_num, 'y': y}, idx),
            X_cat=None,
            idx=idx
        )
    elif X_num.shape[1] == 0: # no numerical
        X_num = None
        _save(
            dataset_dir,
            'my_data',
            task_type,
            **_apply_split({'X_cat': X_cat, 'y': y}, idx),
            X_num=None,
            idx=idx
        )
    else:
        _save(
            dataset_dir,
            'my_data',
            task_type,
            **_apply_split({'X_num': X_num, 'X_cat': X_cat, 'y': y}, idx),
            idx=idx
        )