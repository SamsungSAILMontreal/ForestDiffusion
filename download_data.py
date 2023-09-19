#!/usr/bin/env python
# coding: utf-8

from data_loaders import dataset_loader

if __name__ == "__main__":

    datasets=['iris', 'wine', 'california', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'bean']

    for dataset in datasets:
        X, bin_x, cat_x, int_x, y, bin_y, cat_y, int_y = dataset_loader(dataset)