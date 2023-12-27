import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as skd

from ForestDiffusion import ForestDiffusionModel
from sklearn.utils import check_random_state

rix = 0
rng = check_random_state(rix)
n_upper = 100
n_lower = 100
n = n_upper + n_lower
data, labels = skd.make_moons(
    (n_upper, n_lower), shuffle=False, noise=0.1, random_state=rix)

data4impute = data.copy()
data4impute[:, 1] = np.nan
model = ForestDiffusionModel(
    X=data,
    n_t=100, duplicate_K=100, diffusion_type='vp',
    bin_indexes=[], cat_indexes=[], int_indexes=[], n_jobs=-1)
data_fake = model.generate(batch_size=data.shape[0])

nimp = 1 # number of imputations needed
data_imputefast = model.impute(X=data4impute, k=nimp) # regular (fast)
data_impute = model.impute(X=data4impute, repaint=True, r=10, j=5, k=nimp) # REPAINT (slow, but better)
data_impute2 = model.impute(X=data4impute, repaint=True, r=10, j=1, k=nimp) # REPAINT (slow, but better)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 5));
axes[0, 0].scatter(data[:, 0], data[:, 1]);
axes[0, 0].set_title('original');
axes[0, 1].scatter(data_imputefast[:, 0], data_imputefast[:, 1]);
axes[0, 1].set_title('imputed');
axes[1, 0].scatter(data_impute[:, 0], data_impute[:, 1]);
axes[1, 0].set_title('imputed');
axes[1, 1].scatter(data_impute2[:, 0], data_impute2[:, 1]);
axes[1, 1].set_title('imputed');
plt.tight_layout();
plt.savefig("my_fig.png");