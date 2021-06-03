# Authors: JG, DC
# Date: 6/3/2021
# Purpose: cleans the csv resulting from fuzzy matching
# Filename: A2_post_fuzzy_cleaning.py

import pandas as pd
import numpy as np
import random
import re
import recordlinkage
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Read in the csv from A1_fuzzy Matching
A1_csv = pd.read_csv('my_data/fuzzyMatchResult.csv')
print(A1_csv.head())






