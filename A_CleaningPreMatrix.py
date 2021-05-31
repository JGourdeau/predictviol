#imports
import pandas as pd
import numpy as np
import random
import re
import recordlinkage
import time

# prevent depreciation warnings
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# ## repeated printouts
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

## load in h2a data
h2a = pd.read_excel("./my_data/h2a_2018.xlsx")

## load in investigations/violations data
# url = "../my_data/whd_whisard.csv"
url = "https://enfxfr.dol.gov/data_catalog/WHD/whd_whisard_20210415.csv.zip"
investigations = pd.read_csv(url,
                      index_col=None,
                      dtype={7:'str'})

## convert the dates to datetime objects
investigations['findings_start_date'] = pd.to_datetime(investigations['findings_start_date'], errors='coerce')
investigations['findings_end_date'] = pd.to_datetime(investigations['findings_end_date'], errors = "coerce")

print("h2a")
h2a.head()
h2a.columns
print('investigations')
investigations.head()

## this function will pull out the certification status from a given h2a application
def find_status(one):
    string_version = str(one)                    ## convert to string
    pattern = r'\-\s(.*)$'                       ## define regex pattern
    found = re.findall(pattern, string_version)  ## search for pattern and return what's found
    return found[0]

h2a["status"] = [find_status(one) for one in h2a.CASE_STATUS]   ## put the status in a new column

## filter to applications that have received certification or partial certification
approved_only = h2a.loc[((h2a.status == "CERTIFICATION") | (h2a.status == "PARTIAL CERTIFICATION")),:].copy()

## this function will clean the EMPLOYER_NAME in approved_only (h2a apps) and legal_name in violations (WHD data)
def clean_names(one):
    string_version = str(one)               ## convert to string
    upper_only = string_version.upper()     ## convert to uppercase
    pattern = r"(LLC|CO|INC)\."             ## locate the LLC, CO, or INC that are followed by a period
    replacement = r'\1'                     ## replace the whole pattern with the LLC/CO/INC component
    res = re.sub(pattern, replacement, upper_only)  ## compute and return the result
    return res

## make new "name" columns for the cleaned versions of the names
approved_only["name"] = [clean_names(one) for one in approved_only.EMPLOYER_NAME]
approved_only_pure = approved_only.copy()
investigations["name"] = [clean_names(one) for one in investigations.legal_name]
investigations_cleaned = investigations.loc[investigations.name != "NAN",:].copy()      ## get rid of NAN names

investigations_cleaned[["name","h2a_violtn_cnt"]]
# investigations_cleaned.head()

# subset to just those which have violations
# violations = investigations_cleaned.loc[investigations_cleaned.h2a_violtn_cnt > 0, :].copy()

## violations is now all investigations
violations = investigations_cleaned.copy()

def fuzzyMatch(dbase1, dbase2, blockLeft, blockRight, matchVar1, matchVar2, distFunction, threshold, colsLeft, colsRight):
    link_jobs_debar = recordlinkage.Index() ## initialize our Index
    link_jobs_debar.block(left_on = blockLeft, right_on = blockRight)         ## block on the given block variable

    ## form our index with the two given databases
    candidate_links = link_jobs_debar.index(dbase1, dbase2)

    compare = recordlinkage.Compare()       ## initialize our compare class
    if (len(matchVar1) != len(matchVar2)):  ## ensure matching num. of matching vars
        print("Need to pass in your matching variables in an array and you need to have the same number of matching variables. Please try again. ")
        return

    for i in range(len(matchVar1)):         ## for each matching pair, add to our comparator
        compare.string(matchVar1[i], matchVar2[i], method = distFunction, threshold = threshold)

    compare_vectors = compare.compute(candidate_links, dbase1, dbase2) ## compute
    compare_vectors

    # rename columns
    temp_array = []
    for i in range(len(matchVar1)):
        colName = str(matchVar1[i])
        temp_array.append(colName)
    compare_vectors.columns = temp_array

    ## Find the correct selection
    conditions = []
    for one in matchVar1:
        condition_string = "({one_input} == 1)".format(one_input = one)
        conditions.append(condition_string)
    if (len(conditions) > 1):
        comparison = "&".join(conditions)
    else:
        comparison = conditions[0]
    selected = compare_vectors.query(comparison).copy()

    # Extract index from selection
    n = selected.shape[0]
    index_dbase1_values = []
    index_dbase2_values = []
    for i in range(n):
        index = selected.index[i]
        index_dbase1_values.append(index[0])
        index_dbase2_values.append(index[1])
    selected["index_dbase1"] = index_dbase1_values.copy()
    selected["index_dbase2"] = index_dbase2_values.copy()

    # merge jobs with original columns
    ## this will throw an error if jobs is not the left
    dbase1["index_dbase1"] = dbase1.index
    dbase1_columns = colsLeft
    m1 = pd.merge(selected, dbase1[dbase1_columns], on = "index_dbase1", how = "inner")

    # merge debar with original columns
    dbase2["index_dbase2"] = dbase2.index
    dbase2_columns = colsRight
    m2 = pd.merge(m1, dbase2[dbase2_columns], on = "index_dbase2", how = "inner", suffixes = ["_left", "_right"])

    return m2

#################################################################################################
approved_only["city"] = [one.upper() for one in approved_only.EMPLOYER_CITY]
violations["city"] = [one.upper() for one in violations.cty_nm]

#################################################################################################
blockLeft = "EMPLOYER_STATE"
blockRight = "st_cd"
matchingVarsLeft = ["name","city"]
matchingVarsRight = ["name","city"]
colsLeft = ["status","JOB_START_DATE","JOB_END_DATE","EMPLOYER_STATE", "name","index_dbase1","city"]
colsRight = ["st_cd", "name", "h2a_violtn_cnt","findings_start_date","findings_end_date","index_dbase2","city","ld_dt"]

res = fuzzyMatch(approved_only, violations, blockLeft,blockRight,matchingVarsLeft,matchingVarsRight,"jarowinkler",0.85,colsLeft,colsRight)

fuzzy_match_violations = res.loc[(res.ld_dt >= res.JOB_START_DATE),:].copy()
fuzzy_match_violations

print('we found %s unique employers in the 2018 H2A with violations' %res.name.nunique())

# Make a classifier for Y if the name in the H2A was fuzzy matched in m2
approved_only_pure["is_violator"] = np.where(approved_only_pure.name.isin(list(fuzzy_match_violations.name_y)), 1, 0)
approved_only_pure.is_violator.value_counts()
#approved_only_pure.head()

approved_only_pure.dtypes

print("there are %s applications in the H2A approved Dataset" %len(approved_only_pure))
print('but only %s unique companies within those applications' %approved_only_pure.name.nunique())

def find_pattern(string, pat):
    res = re.findall(pat, string)
    if (len(res) > 0):
        return True
    else:
        return False

def find_mode(col_of_interest):
    list_version = list(col_of_interest.values)
    values = sorted(list_version, key = list_version.count, reverse=True)  ## sorted the adj_only list in descending order
    values_no_dups = list(dict.fromkeys(values))                      ## remove duplicates while preserving order to get top 5
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
    return res

test_res = form_representative(approved_only_pure, "name")
test_res.head()


test_res.to_csv("repMatrix.csv")

test_res.shape
