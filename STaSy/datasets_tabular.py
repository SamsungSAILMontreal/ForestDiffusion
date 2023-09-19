import json
import logging
import os
import urllib

import numpy as np

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

LOGGER = logging.getLogger(__name__)

def load_data(numpy_data, categorical_columns, ordinal_columns): 
    return numpy_data, (categorical_columns, ordinal_columns)
