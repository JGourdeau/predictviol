

############################## Import packages

import os
from os import listdir
import csv
import itertools
import numpy as np
from numpy import loadtxt
import pandas as pd
from numpy import genfromtxt
import scipy
import sklearn
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier



############################## Read in data

dir_fake = "fake_dir"

## read in .csv of features, folds, and labels
labels = pd.read_csv(dir_fake)
features = pd.read_csv(dir_fake)



############################ Subset to common app and university supp separately
## separate into common app and University supplement
features_lowerthres_CA = features_lowerthres.loc[features_lowerthres.common_app == 1, ].copy().sort_values(by = 
                                                                        "Research_ID")
features_CA = features.loc[features.common_app == 1, ].copy().sort_values(by = 
                                                                        "Research_ID")
features_lowerthres_supp = features_lowerthres.loc[features_lowerthres.common_app == 0, ].copy().sort_values(by = 
                                                                        "Research_ID")
features_supp = features.loc[features.common_app == 0, ].copy().sort_values(by = 
                                                                        "Research_ID")


## summarize
print("There are " + str(features_lowerthres_CA.shape[0]) + " common app essays")
print("There are " + str(features_lowerthres_supp.shape[0]) + " univ supplement essays")

## subset labels to one appearance of each research id (was previously repeated long form)
labels = labels.sort_values(by = "Research_ID").drop_duplicates(keep = "first")


############################## Functions

def get_features_labels(ids, features, labels, feature_type):
    
    feature_cols = list(set(features.columns).difference(['Research_ID', 'fold']))
    
    ## text features (two rows per ID due to diff essays)
    if feature_type ==  "text":
        label_nonarray = labels.loc[labels.Research_ID.isin(ids)]
        label = np.array(label_nonarray[['SB_invite']])
        feature_cols = [col for col in feature_cols if "term" in col or 
                       "common_app" in col]
        features = np.array(features.loc[features.Research_ID.isin(ids), feature_cols])
        ids_toreturn = label_nonarray.Research_ID
        
        print("Dimensions of text feature matrix:" + str(features.shape))
        
    ## non-text features (one row per ID bc diff essays don't matter)
    elif feature_type == "nontext":
        
        ## deduplicate labels and make array
        label_non_duplicated = labels.loc[labels.Research_ID.isin(ids)].drop_duplicates(subset = "Research_ID")
        label = np.array(label_non_duplicated[['SB_invite']])
        
        ## deduplicate features and make array
        feature_cols = [col for col in feature_cols if "term" not in col and
                       "common_app" not in col]
        features_non_duplicated = features[features.Research_ID.isin(ids)].drop_duplicates(subset = "Research_ID")
        features = np.array(features_non_duplicated[feature_cols])
        ids_toreturn = label_non_duplicated.Research_ID
        print("Dimensions of non-text feature matrix:" + str(features.shape))
        
    ## both types of features 
    elif feature_type == "both":
        label_nonarray = labels.loc[labels.Research_ID.isin(ids)]
        label = np.array(label_nonarray[['SB_invite']])
        features = np.array(features.loc[features.Research_ID.isin(ids), feature_cols])
        ids_toreturn = label_nonarray.Research_ID
        print("Dimensions of non-text and text feature matrix:" + str(features.shape))
        
    return(label, features, ids_toreturn)
   
    
  

def evaluate_models(y_pred, label_test):
    all_results = precision_recall_fscore_support(label_test,
                                y_pred)
    
    all_results_1 = [i[0] for i in all_results][0:3]
    return(all_results_1)


## function to estimate model in one fold
def estimate_models(model_list, names_list, features, labels, feature_type):

    evals_df = {}
    store_pred_allmodels = {}
    for j in range(0, len(model_list)):

        ## pull out model
        one_model = model_list[j]
        
        print("fitting model: " + str(one_model))

        ## iterate over folds to estimate and evaluate
        store_evals_fold = []
        store_pred_allfolds = []
        for i in range(1, 6):
            
            ## ids for fold
            which_fold = [i]
            train_folds = list(set(list(range(1, 6))).difference(which_fold))
            train_ids = features.Research_ID[features.fold.isin(train_folds)]
            test_ids = features.Research_ID[features.fold.isin(which_fold)]
            
            ## label and features
            
            (label_train, training_features, train_final_ids) = get_features_labels(train_ids, features, labels,
                                                                  feature_type)
            (label_test, test_features, test_final_ids) = get_features_labels(test_ids, features, labels, feature_type)
            
            ## fit the model and evaluate
            print("estimating for fold:" + str(i))
            one_model.fit(training_features, label_train)
            print("estimated model")
            y_pred = one_model.predict(test_features)
            y_score = one_model.predict_proba(test_features)[:, 1]
            print("generated predictions")
            
            ## store predictions
            store_pred_allfolds.append(pd.DataFrame({'Research_ID': test_final_ids,
                               'Model': names_list[j],
                                'Binary_pred': y_pred,
                                'Score': y_score, 
                                'Observed_label': label_test.tolist()}))
            
            ## store evaluations
            store_evals_fold.append(evaluate_models(y_pred, label_test))
            evals_df[names_list[j]] = np.mean(store_evals_fold, 0)
            store_pred_allmodels[names_list[j]] = pd.concat(store_pred_allfolds)
            
    return(evals_df, store_pred_allmodels)

def evalsarray_to_df(eval_array, model_name):
  eval_df= pd.DataFrame.from_dict(eval_array, orient = "index")
  eval_df.columns = accuracy_metrics
  eval_df['type'] = model_name
  eval_df['model'] = eval_df.index
  return(eval_df)

def predict_to_df(predictions, model_name):
  pred_df = pd.concat(predictions).reset_index()
  pred_df['type'] = model_name
  return(pred_df)

############################## Full classifiers

## create a list of model objects
classifiers_list = [DecisionTreeClassifier(random_state=0, max_depth = 5), 
                    DecisionTreeClassifier(random_state=0, max_depth = 50), 
                    RandomForestClassifier(n_estimators = 100, max_depth = 20),
                    RandomForestClassifier(n_estimators = 1000, max_depth = 20),
                    GradientBoostingClassifier(criterion='friedman_mse', n_estimators=100),
                    GradientBoostingClassifier(criterion='friedman_mse', n_estimators=1000),
                AdaBoostClassifier(), 
                LogisticRegression(),
                LogisticRegressionCV(),
                LogisticRegression(penalty = "l1"),
                LogisticRegressionCV(solver = "liblinear", 
                                 penalty = "l1")]
print("Length of classifier list is:" + str(len(classifiers_list)))
names_list = ['dt_shallow', 'dt_deep',
              'rf_few', 'rf_many',
              'gb_few', 'gb_many',
              'ada', 
              'logit', 'logitcv', 'logitl1',
              'logitl1cv']

print("Length of classifier list is:" + str(len(names_list)))

##############################  Test classifiers
#classifiers_list= [DecisionTreeClassifier(random_state=0), RandomForestClassifier(random_state = 0)]
#names_list = ['dt', 'rf']


############################## Estimate models on test classifiers

(evals_df_onlytext_lowerthres_CA, store_pred_onlytext_lowerthres_CA) = estimate_models(classifiers_list,
                          names_list,
                          features_lowerthres_CA,
                          labels,
                        feature_type = "text")


(evals_df_onlytext_lowerthres_supp, store_pred_onlytext_lowerthres_supp) = estimate_models(classifiers_list,
                          names_list,
                          features_lowerthres_supp,
                          labels,
                        feature_type = "text")

(evals_df_onlytext_higherthres_CA, store_pred_onlytext_higherthres_CA) = estimate_models(classifiers_list,
                          names_list,
                          features_CA,
                          labels,
                        feature_type = "text")

(evals_df_onlytext_higherthres_supp, store_pred_onlytext_higherthres_supp) = estimate_models(classifiers_list,
                          names_list,
                          features_supp,
                          labels,
                        feature_type = "text")



############################## Combine evals into one dataframe

accuracy_metrics = ['precision', 'recall', 'F1Score']
eval1 = evalsarray_to_df(evals_df_onlytext_lowerthres_CA,
                        "onlytext_lowerthres_CA")
eval2 = evalsarray_to_df(evals_df_onlytext_higherthres_CA,
                        "onlytext_higherthres_CA")
eval3 = evalsarray_to_df(evals_df_onlytext_lowerthres_supp,
                        "onlytext_lowerthres_supp")
eval4 = evalsarray_to_df(evals_df_onlytext_higherthres_supp,
                        "onlytext_higherthres_supp")

## combine evals and write to csv
all_evals = pd.concat([eval1, eval2, eval3, eval4])
all_evals.to_csv(fake_dir,
                  index = False)


## combine predictions and write to csv
pred1 = predict_to_df(store_pred_onlytext_lowerthres_CA, "onlytext_lowerthres_CA")
pred2 = predict_to_df(store_pred_onlytext_higherthres_CA, "onlytext_higherthres_CA")
pred3 = predict_to_df(store_pred_onlytext_lowerthres_supp, "onlytext_lowerthres_supp")
pred4 = predict_to_df(store_pred_onlytext_higherthres_supp, "onlytext_higherthres_supp")


all_pred = pd.concat([pred1, pred2, pred3, pred4])
all_pred.to_csv(fake_dir,
                  index = False)



print("finished script")
