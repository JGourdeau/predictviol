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

def find_pattern(string, pat):
    res = re.findall(pat, string)
    if (len(res) > 0):
        return True
    else:
        return False

def find_mode(col_of_interest):
    list_version = list(col_of_interest.values)
    values = sorted(list_version, key = list_version.count, reverse=True)  ## sorted the adj_only list in descending order
    values_no_dups = list(dict.fromkeys(values))                        ## remove duplicates while preserving order to get top 5
    return values_no_dups[0]


def form_representative(df, col_to_groupby):
    print('**** FORMING REPS ****')
    list_of_reps = []
    for one in df[col_to_groupby].unique():
        temp_df = df.loc[df[col_to_groupby] == one].copy()
        to_add = {}
        for col in temp_df:
            col_type = df.dtypes[col]
            if (col_type == "int64"):
                to_add[col] = temp_df[col].mean(skipna = True)
            elif (col_type == "object"):
                to_add[col] = find_mode(temp_df[col])
            elif (col_type == "float64"):
                to_add[col] = temp_df[col].mean(skipna = True)
            elif (col_type == "datetime64[ns]"):
                if (find_pattern(str(col),r'START')):
                    to_add[col] = temp_df[col].min()
                elif (find_pattern(str(col),r'END')):
                    to_add[col] = temp_df[col].max()
                else:
                    to_add[col] = temp_df[col].min()
            else:
                print("Other type")
        list_of_reps.append(to_add)

    res = pd.DataFrame(list_of_reps)
    print("**** DONE FORMING REPS *****")
    return res


# Read in the csv from A1_fuzzy Matching
res = pd.read_csv('./my_data/fuzzyMatchResult.csv')
print(res.head())

approved_only_pure = pd.read_csv('./approvedOnly.csv')

res["load_date_cleaned"] =  pd.to_datetime(res['ld_dt'], errors='coerce')
res["JOB_START_DATE"] =  pd.to_datetime(res['JOB_START_DATE'], errors='coerce')

# import pytz

# utc=pytz.UTC
res["load_date_cleaned"] = res["load_date_cleaned"].apply(lambda d: d.replace(tzinfo=None))
res["JOB_START_DATE"] = res["JOB_START_DATE"].apply(lambda d: d.replace(tzinfo=None))
# res["load_date_cleaned"] = res.load_date_cleaned.replace(tzinfo=utc)
# res["JOB_START_DATE"] = res.JOB_START_DATE.replace(tzinfo=utc)

fuzzy_match_violations = res.loc[(res.load_date_cleaned >= res.JOB_START_DATE),:].copy()
fuzzy_match_violations

print('we found %s unique employers in the 2018 H2A with violations' %res.name.nunique())
# Make a classifier for Y if the name in the H2A was fuzzy matched in m2
approved_only_pure["is_violator"] = np.where(approved_only_pure.name.isin(list(fuzzy_match_violations.name_y)), 1, 0)
approved_only_pure.is_violator.value_counts()
#approved_only_pure.head()

approved_only_pure.dtypes

print("there are %s applications in the H2A approved Dataset" %len(approved_only_pure))
print('but only %s unique companies within those applications' %approved_only_pure.name.nunique())

test_res = form_representative(approved_only_pure, "name")
test_res.head()


test_res.to_csv("repMatrix2.csv")

test_res.shape






