# predictviol

## Project Overview
Final project for QSS 20 - The aim of this project is to use the features within the DOL quarterly Jobs data (considered the universe of employers) to predict whether an H2A employer will be investigated for a violation or be found to have violated worker's rights as listed within the WHD Violations Data. We will be using SKlearn for feature extraction and ML implimentation. 

## Data Sources 
* WHD Violations Data: https://enfxfr.dol.gov/data_catalog/WHD/whd_whisard_20210415.csv.zip
  * File structure: [/data/whd_data_dictionary.csv](https://github.com/JGourdeau/predictviol/blob/main/data/whd_data_dictionary.csv) or [rebeccajohnson88
/qss20_s21_proj](https://github.com/rebeccajohnson88/qss20_s21_proj/tree/main/data/documentation)

* DOL Quarterly Jobs Data: https://www.dol.gov/agencies/eta/foreign-labor/performance
  * File structure: [/data/H-2A_FY18_Record_Layout](https://github.com/JGourdeau/predictviol/blob/main/data/H-2A_FY18_Record_Layout.pdf) or [www.dol.gov](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/H-2A_FY18_Record_Layout.pdf)
  
## Methods

1.) Data Cleaning:
* [A1_fuzzy_matching.py](https://github.com/JGourdeau/predictviol/blob/main/A1_fuzzy_matching.py)
* Takes in:
    - 2018 DOL Quarterly Jobs Data
    - WHD Violations Data
* Does: 
    - Creates a case status for each application
    - Filters to applications which have received certification or partial certification
    - Cleans application and violation employer and city names to improve matching performance
    - Fuzzy Matches H2A applications from DOL to investigations from WHD
    - Saves a csv of approved H2A applications (ApprovedOnly.csv)
    - Saves a csv of fuzzy matches between approved applications and investigations
* Outputs: 
    - A csv of fuzzy matches between approved applications and investigations (fuzzyMatchResult.csv)
    - A csv of approved H2A applications (ApprovedOnly.csv)

2.) Additional Cleaning: 
* [A2_post_fuzzy_cleaning.py](https://github.com/JGourdeau/predictviol/blob/main/A1_post_fuzzy_cleaning.py)
* Takes in:
    - FuzzyMatchResult.csv
    - ApprovedOnly.csv 
* Does: 
    - Allows user to set a mode either 'predict_violations' or 'predict_investigations'. 
    - If predict_violation mode selected: subsets the fuzzy matches to only those with a violation count. 
    - Subsets the fuzzy matches to investigation/violation load dates which are after the job start date in the application. 
    - creates a classifier for each application in the approved applications based on a companies presence in the violations/investigations fuzzy matches. 
    - prints out unique number of applications, companies and the violation counts. 
    - Creates one "representative application" for each of the unique companies in the application data set as each company may have more than one application. 
    - Binds these representative applications into a new representative data frame. 
* Outputs: 
    - Depending on the mode: 
        - if predict_violations: 
            - outputs a csv the representative applications with a classifier corresponding to those companies found to have committed a violation.
            - (repMatrixforpredict_violations.csv) 
        - if predict_investigations: 
            - outputs a csv the representative applications with a classifier corresponding to those companies which have been investigated. 
            - (repMatrixforpredict_investigations.csv)

2.) Feature Matrix Preparation and Model Fitting: 
* [B_feature_matrix_prep.ipynb](https://github.com/JGourdeau/predictviol/blob/main/JGWorking/B_feature_matrix_prep.ipynb)
* Takes in either from A1_post_fuzzy_cleaning: 
    - repMatrixforpredict_violations.csv
    - repMatrixforpredict_investigations.csv
* Does: 
    - Reads in the data and drops (1) columns with only null values
    - Automatically separates columns (aka 'features') into numeric and categorical features
    - Imputes the numeric and categorical features separately
    - Remerges the imputed data and uses OneHot Encoding to prepare for model 
    - Applies an 80/20 train test split to create a training and testing dataset 
    - Using a random forest classifier, fits a model to the data
    - Generates a confusion matrix to visually inspect performance
* Outputs: 
    - prints a confusion matrix and statistics related to model accuracy
  




