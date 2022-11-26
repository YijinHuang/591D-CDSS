import os, glob
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE


def concat_features(df_feat, df):
    return pd.concat([df_feat, df], axis=1)

def add_pct_features(df):
    df_pct = df.groupby('patient').pct_change().iloc[:,1:-3]
    df_pct.columns = [str(col) + '_pct' for col in df_pct.columns]
    return df_pct

def add_diff_features(df):
    df_diff = df.groupby('patient').diff().iloc[:,1:-3]
    df_diff.columns = [str(col) + '_diff' for col in df_diff.columns]
    return df_diff

def add_lag_features(df):
    df_lag = df.groupby('patient').shift(1).iloc[:,1:-3]
    df_lag.columns = [str(col) + '_lag' for col in df_lag.columns]
    return df_lag

def add_cummax_features(df):
    df_cummax = df.groupby('patient').cummax().iloc[:,1:-3]
    df_cummax.columns = [str(col) + '_cummax' for col in df_cummax.columns]
    return df_cummax

def add_polynomial_features(df):
    df_poly = df.groupby('patient').transform(lambda s: s**2).iloc[:,1:-3]
    df_poly.columns = [str(col) + '_poly' for col in df_poly.columns]
    return df_poly

def add_cummin_features(df):
    df_cummin = df.groupby('patient').cummin().iloc[:,1:-3]
    df_cummin.columns = [str(col) + '_cummin' for col in df_cummin.columns]
    return df_cummin

def add_rolling_mean_features(df, windows=5):
    df_roll = df.groupby('patient').transform(lambda s: s.rolling(windows, min_periods=1).mean()).iloc[:,1:-3]
    df_roll.columns = [str(col) + '_roll_mean' for col in df_roll.columns]
    return df_roll

def add_rolling_std_features(df, windows=5):
    df_roll = df.groupby('patient').transform(lambda s: s.rolling(windows, min_periods=1).std()).iloc[:,1:-3]
    df_roll.columns = [str(col) + '_roll_std' for col in df_roll.columns]
    return df_roll

def add_rolling_max_features(df, windows=5):
    df_roll = df.groupby('patient').transform(lambda s: s.rolling(windows, min_periods=1).max()).iloc[:,1:-3]
    df_roll.columns = [str(col) + '_roll_max' for col in df_roll.columns]
    return df_roll

def add_rolling_min_features(df, windows=5):
    df_roll = df.groupby('patient').transform(lambda s: s.rolling(windows, min_periods=1).min()).iloc[:,1:-3]
    df_roll.columns = [str(col) + '_roll_min' for col in df_roll.columns]
    return df_roll

def add_rolling_var_features(df, windows=5):
    df_roll = df.groupby('patient').transform(lambda s: s.rolling(windows, min_periods=1).var()).iloc[:,1:-3]
    df_roll.columns = [str(col) + '_roll_var' for col in df_roll.columns]
    return df_roll


## Load all the data so we can quickly combine it and explore it. 
pfile = 'CinC.pickle'
pfile_test = 'CinC_test.pickle'
if os.path.isfile(pfile):
  CINCdat = pd.read_pickle(pfile)
else:
  os.chdir("../training_2022-11-13")
  extension = 'csv'
  all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
  CINCdat = pd.concat([pd.read_csv(f).assign(patient=os.path.basename(f).split('.')[0]) for f in all_filenames ])
  os.chdir(os.path.dirname(__file__))
  CINCdat.to_pickle(pfile)
print(len(CINCdat)) # should be n=197233

if os.path.isfile(pfile_test):
  CINCdat_test = pd.read_pickle(pfile_test)
else:
  os.chdir("../testing_2022-11-13")
  extension = 'csv'
  all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
  CINCdat_test = pd.concat([pd.read_csv(f).assign(patient=os.path.basename(f).split('.')[0]) for f in all_filenames ])
  os.chdir(os.path.dirname(__file__))
  CINCdat_test.to_pickle(pfile_test)
print(len(CINCdat_test)) # should be n=40442

## Forward-fill missing values
CINCdat.update(CINCdat.groupby('patient').ffill())
CINCdat_test.update(CINCdat_test.groupby('patient').ffill())

## advanced missing values filling
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# imputer = IterativeImputer(max_iter=10, random_state=0)
# imputer.fit(CINCdat.iloc[:,0:-3])
# CINCdat.iloc[:,0:-3] = imputer.transform(CINCdat.iloc[:,0:-3])
# CINCdat_test.iloc[:,0:-3] = imputer.transform(CINCdat_test.iloc[:,0:-3])

## Get reference ranges for variables using only non-sepsis patients as 'normal'
CINCdat_NOsepsis = CINCdat[~CINCdat.patient.isin(np.unique(CINCdat.patient[CINCdat.SepsisLabel==1]))]
CINCdat_NOsepsis = CINCdat_NOsepsis[CINCdat_NOsepsis.ICULOS>1]
CINCdat_NOsepsis.drop(['patient','SepsisLabel','Sex'],axis=1,inplace=True)
meanCINCdat = round(CINCdat_NOsepsis.mean(axis=0),2)
sdCINCdat = round(CINCdat_NOsepsis.std(axis=0),2)
np.set_printoptions(suppress=True)
print('x_mean = np.array(')
print(np.array(meanCINCdat),')')
print('x_std = np.array(')
print(np.array(sdCINCdat),')')

## Obtain the z-scores for all the variables
CINCdat_zScores = CINCdat
CINCdat_test_zScores = CINCdat_test
cols = CINCdat_zScores.columns.drop(['patient','SepsisLabel','Sex'])
for c in cols:
  CINCdat_zScores[c] = (CINCdat_zScores[c]-meanCINCdat[c])/sdCINCdat[c]
  CINCdat_test_zScores[c] = (CINCdat_test_zScores[c]-meanCINCdat[c])/sdCINCdat[c]


## Add features
processed_CINCdat = CINCdat_zScores.copy()
processed_CINCdat_test = CINCdat_test_zScores.copy()

CINCdat_zScores_roll_std = add_rolling_std_features(CINCdat_zScores, windows=10)
CINCdat_test_zScores_roll_std = add_rolling_std_features(CINCdat_test_zScores, windows=10)
processed_CINCdat = concat_features(CINCdat_zScores_roll_std, processed_CINCdat)
processed_CINCdat_test = concat_features(CINCdat_test_zScores_roll_std, processed_CINCdat_test)

CINCdat_zScores_cummax = add_cummax_features(CINCdat_zScores)
CINCdat_test_zScores_cummax = add_cummax_features(CINCdat_test_zScores)
processed_CINCdat = concat_features(CINCdat_zScores_cummax, processed_CINCdat)
processed_CINCdat_test = concat_features(CINCdat_test_zScores_cummax, processed_CINCdat_test)

CINCdat_zScores_cummin = add_cummin_features(CINCdat_zScores)
CINCdat_test_zScores_cummin = add_cummin_features(CINCdat_test_zScores)
processed_CINCdat = concat_features(CINCdat_zScores_cummin, processed_CINCdat)
processed_CINCdat_test = concat_features(CINCdat_test_zScores_cummin, processed_CINCdat_test)

## Replace values still missing with the mean
processed_CINCdat = processed_CINCdat.fillna(0)
processed_CINCdat_test = processed_CINCdat_test.fillna(0)

## fill infinities with 1
processed_CINCdat = processed_CINCdat.replace([np.inf, -np.inf], 0)
processed_CINCdat_test = processed_CINCdat_test.replace([np.inf, -np.inf], 0)

## Save the data
processed_CINCdat.to_pickle('processed_CinC.pickle')
processed_CINCdat_test.to_pickle('processed_CinC_test.pickle')

## feature selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
lreg = LogisticRegression(random_state=0, max_iter=1000)
selector = SelectFromModel(estimator=lreg).fit(processed_CINCdat.iloc[:, 0:-3], processed_CINCdat.SepsisLabel)
idx = selector.get_support()
selected_features = processed_CINCdat.columns[0:-3][idx]
selected_CINCdat = pd.concat([processed_CINCdat[selected_features], processed_CINCdat.iloc[:,-3:]], axis=1)
selected_CINCdat_test = pd.concat([processed_CINCdat_test[selected_features], processed_CINCdat_test.iloc[:,-3:]], axis=1)
print(selected_features)

## Save the data
selected_CINCdat.to_pickle('selected_CINC.pickle')
selected_CINCdat_test.to_pickle('selected_CINC_test.pickle')

