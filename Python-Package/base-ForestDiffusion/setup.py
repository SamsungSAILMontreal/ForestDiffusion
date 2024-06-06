from setuptools import setup, find_packages

VERSION = '1.0.6' 
DESCRIPTION = 'Generating and Imputing Tabular Data via Diffusion and Flow XGBoost Models'
LONG_DESCRIPTION = 'Tabular data is hard to acquire and is subject to missing values. This paper proposes a novel approach to generate and impute mixed-type (continuous and categorical) tabular data using score-based diffusion and conditional flow matching. Contrary to previous work that relies on neural networks as function approximators, we instead utilize XGBoost, a popular Gradient-Boosted Tree (GBT) method. In addition to being elegant, we empirically show on various datasets that our method i) generates highly realistic synthetic data when the training dataset is either clean or tainted by missing data and ii) generates diverse plausible data imputations. Our method often outperforms deep-learning generation methods and can trained in parallel using CPUs without the need for a GPU. To make it easily accessible, we release our code through a Python library and an R package <arXiv:2309.09968>.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="ForestDiffusion", 
        version=VERSION,
        author="Alexia Jolicoeur-Martineau",
        author_email="<alexia.jolicoeur-martineau@mail.mcgill.ca>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        url='https://github.com/SamsungSAILMontreal/ForestDiffusion',
        install_requires=['numpy', 'scikit-learn', 'xgboost>=2.0.0', 'lightgbm', 'catboost', 'pandas'], 
        keywords=['python', 'AI', 'xgboost', 'GBT', 'tree', 'forest', 'tabular', 'diffusion', 'flow'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)