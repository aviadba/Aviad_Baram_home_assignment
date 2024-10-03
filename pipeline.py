"""
This module implements the pipeline for processing home assignment data from nVidia

Build a classifier to predict Target (tes pass/fail).
Goal: Achieve a high recall for test fails (actual tests are expensive)
"""

# module imports
import pandas as pd
from pipe import Pipe
from visualizations import *

GENERATE_PLOTS = False  # set to True in order to generate report plots throughout the pipeline
VERBOSE = True  # Set to False for a silent run (confusion matrices and scores are always printed to stdout)
SAVE_INTERMEDIATE = False  # set to True to save X_train, y_train, X_test and y_test before model training

# import the raw data as pandas DataFrame
raw_data_path = r'/home/aviad/Projects/nvidia_data/home_assignment.feather'
df = pd.read_feather(raw_data_path, columns=None, use_threads=True)
if GENERATE_PLOTS:
    showRawDataDtype(df)
# create solution e2e pipe
pipe = Pipe()
# preprocess raw data
X, y = pipe.rawDataPreprocess(df, verbose=VERBOSE)
if GENERATE_PLOTS:
    showImbalance(y)
# split raw data into train and test sets
X_train, X_test, y_train, y_test = pipe.imbalance_split_train_test(X, y, verbose=VERBOSE)
if SAVE_INTERMEDIATE:
    root_path = r'data/'
    for fn, dt in zip(['X_train', 'X_test', 'y_train', 'y_test'], [X_train, X_test, y_train, y_test]):
        filepath = root_path+fn+'.csv'
        np.savetxt(filepath, dt, delimiter=",")
# reduce number of dimension using PCA
X_train_pca, var_cumsum, pca_num = pipe.pca_fit_transform(X_train, verbose=VERBOSE)
if GENERATE_PLOTS:
    plotPCAvaraince(var_cumsum, pca_num)
# train the model
pipe.train(X_train_pca, y_train)
# predict training data
y_train_hat = pipe.predict(X_train)
cm_train, prec_train, rec_train, f1_train = pipe.evaluate(y_train, y_train_hat)
print('Train confusion matrix:')
print(cm_train)
print(f'Precision: {prec_train:.2f}, Recall: {rec_train:.2f}, F1: {f1_train:.2f}')
# test the model
y_test_hat = pipe.predict(X_test)
cm_test, prec_test, rec_test, f1_test = pipe.evaluate(y_test, y_test_hat)
print('Test confusion matrix:')
print(cm_test)
print(f'Precision: {prec_test:.2f}, Recall: {rec_test:.2}, F1: {f1_test:.2}')



