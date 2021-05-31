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
* [A_CleaningPreMatrixPrep.ipynb](https://github.com/JGourdeau/predictviol/blob/main/JGWorking/A_CleaningPreMatrixPrep.ipynb)
* Takes in:
    - 2018 DOL Quarterly Jobs Data
    - WHD Violations Data
* Does: 
    - Cleans application and violation employer names to improve matching performance
    - Fuzzy Matches H2A applications from DOL to Violations from WHD
    - Uses these matches, creates a binary classifier for violators or non-violators within the applications Data
    - Creates one "representative application" for each of the unique companies in the application data set
    - Binds these representative applications into a new representative data frame
* Outputs: 
    - A CSV, RepMatrix.csv, for use with notebook/script B

2.) Feature Matrix Preparation: 
* [B_feature_matrix_prep.ipynb](https://github.com/JGourdeau/predictviol/blob/main/JGWorking/B_feature_matrix_prep.ipynb)
* Takes in: 
    - the CSV RepMatrix.csv from [A_CleaningPreMatrixPrep.ipynb](https://github.com/JGourdeau/predictviol/blob/main/JGWorking/A_CleaningPreMatrixPrep.ipynb)
* Does: 
    - Reads in the data and drops columns with all null values
    - Automatically separates columns (aka 'features') into numeric and categorical features
    - Imputes the numeric and categorical features separately
    - Remerges the imputed data and uses OneHot Encoding to prepare for model 
    - Applies an 80/20 train test split to create a training and testing dataset 
    - Using a random forest classifier, fits a model to the data
    - Generates a confusion matrix to visually inspect performance
* Outputs: 
    - None
  
## Results




