"""
This module implements preprocessing, feature selection and transformation, model training and evaluation for
 the home assignment of building a binary classifier for predicting the results of TLJYWBE test.

Full data pipeline is outlined in the file pipline.py

"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE  # for imbalanced datasets
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


class Pipe:
    VARIANCE_TH = 0.95  # variance threshold for principle component selection
    N_FOLD = 3  # Cross validation n-fold

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.scaler1 = None  # Standard Scaler for principal component analysis selection
        self.scaler2 = None  # standard scaler for scaling transformed vectors
        self.pca = None  # Principal Componenet Analysis
        self.n_pca = int  # num of componenets to select from PCA

        self.rfc = None  # Random Forest Classifier


    @staticmethod
    def rawDataPreprocess(df: pd.DataFrame, verbose: bool = False):
        """
        Preliminary preprocessing for raw data.
        Includes:
        mapping response column to boolean pass fail
        sropping all rows with no NaN response
        Droping all column with NaN values
        Selecting only numerical columns
        :param df: pd.DataFrame
            Raw dataframe for processing
        :param verbose: bool
        :return: pd.Dataframe, pd.Series
            numerical predictor as DataFrame, boolean response as Series
        """
        response_feature_name = 'TLJYWBE'
        response_th = 10 ** -5
        if verbose:
            raw_df_shape = df.shape
            pd.set_option('display.max_columns', 10)
            print(f'Raw data: total samples: {raw_df_shape[0]}, total features: {raw_df_shape[1]}')
            print(f'float64 col: {len(df.select_dtypes('float64').columns)}')
            print(f'int64 cols: {len(df.select_dtypes('int64').columns)}')
            print(f'categorical col: {len(df.select_dtypes('object').columns)}')
            print(df.sample(5))
        # map to pass/fail test
        failidx = df.loc[df[response_feature_name] >= response_th].index
        df.loc[failidx, response_feature_name] = 1
        df.loc[~df[response_feature_name].index.isin(failidx), response_feature_name] = 0
        # remove all NaN response rows and change to boolean
        df.dropna(subset=[response_feature_name], inplace=True)
        df[response_feature_name] = df[response_feature_name].astype('bool')
        if verbose:
            tot_test_pass = len(df.loc[df[response_feature_name] == 1].index)
            tot_test_fail = len(df.loc[df[response_feature_name] == 0].index)
            print(f'Total test pass: {tot_test_pass}, total test fail: {tot_test_fail}')
            # check for nulls
            total_nan = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            pct_missing = (total_nan / total_cells) * 100
            print(f'precent missing: {pct_missing:.2f}%')
        # remove all features with nan values - otherwise require imputing introduce bias)
        df.dropna(axis=1, how='any', inplace=True)

        all_columns = df.columns.tolist()  # get all starting columns for pipelinne
        y = df[response_feature_name]
        X = df.drop(columns=response_feature_name)
        X = X.select_dtypes('float64')
        if verbose:
            print(f'Following raw preprocessing:\nFeatures: {X.shape[1]} (numerical)\nSamples: {X.shape[0]}')
        return X, y

    @staticmethod
    def imbalance_split_train_test(X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        """
        Split train-test with Under sampling and over spelling to create equal sized sets
        :param X: DataFrame
            Features dataframe
        :param y: Series
            Target series
        :param verbose: bool
        """
        # use under sampling followed by over sampling - alternatively - use SMOTE
        rush = RandomUnderSampler(random_state=40, replacement=False, sampling_strategy=0.5)
        x_rush, y_rush = rush.fit_resample(X, y)
        # oversampeling of test fail
        rosh = RandomOverSampler(random_state=40, sampling_strategy=1)
        x_hres, y_hres = rosh.fit_resample(x_rush, y_rush)
        # split train/test for imbalanced sets
        X_train, X_test, y_train, y_test = train_test_split(x_hres,
                                                            y_hres,
                                                            test_size=0.2,
                                                            random_state=40)
        if verbose:
            train_pass = (y_train == 1).sum()
            train_fail = (y_train == 0).sum()
            test_pass = (y_test == 1).sum()
            test_fail = (y_test == 0).sum()
            print(f'Post sampling label distribution:\nTrain ({len(y_train)}):\tPASS-{train_pass}\tFAIL-{train_fail}')
            print(f'Test ({len(y_test)}):\tPASS-{test_pass}\tFAIL-{test_fail}')
        return X_train.values, X_test.values, y_train.values, y_test.values

    def pca_fit_transform(self, X: pd.DataFrame, verbose: bool = False):
        """
        Apply scaler and PCA on numerical feature DataFrame
        :param X: pd.DataFrame
            DataFram of numerical features
        :param var_th: float
            Variance threshold to achieve after PCA. This value will determine the number of features used in the rfc
        :param verbos: bool
        :return np.array, np.array
            X data after transforming and slicing the features to hot VARIANCE_TH, cumsum of explained variance after PCA
        """
        self.scaler1 = StandardScaler()
        X_train_scaled = self.scaler1.fit_transform(X)
        self.pca = PCA()
        X_pca = self.pca.fit_transform(X_train_scaled)
        total_explained_variance = self.pca.explained_variance_ratio_.cumsum()
        n_over_th = len(total_explained_variance[total_explained_variance >= self.VARIANCE_TH])
        self.n_pca = X_train_scaled.shape[1] - n_over_th + 1
        if verbose:
            print(
                f'numerical features projected to PCA keeping {self.n_pca} PCA features for {self.VARIANCE_TH * 100}% variance')
        self.scaler2 = StandardScaler()
        X_train_scaled2 = self.scaler2.fit_transform(X_pca[:, :self.n_pca])
        return X_train_scaled2, total_explained_variance, self.n_pca

    # Modeling using Random Forest Classifier
    def train(self, X_train, y_train):
        """
        Wrapper for training RFC model by cross validation
        :param X_train: np.array
        :param y_train: np.array
        :return:
        """
        # select optimized random forest n_estimators and depth using CV grid search
        v_rfc = self._crossValidate(X_train, y_train)
        self.rfc = v_rfc.fit(X_train, y_train)

    def _crossValidate(self, X_train, y_train):
        """
        Carry out  a 3 way cross validation for model parameters n_estimators and tree_depth
        :param X_train: np.array
        :param y_train: np.array
        """
        n_estimators = [100, 200, 300]
        max_depth = [2, 3, 7, 10]
        min_samples_split = [2, 5, 10]
        param_grid = {'n_estimators': n_estimators,
                      'min_samples_split': min_samples_split,
                      'max_depth': max_depth}
        rfc = RandomForestClassifier(class_weight='balanced')
        gs = GridSearchCV(rfc, param_grid, cv=self.N_FOLD, verbose=1, scoring='recall', n_jobs=-1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_

    def predict(self, X: pd.DataFrame):
        """
        Predict
        :param X: pd.Dataframe
            Feature dataframe
        :return: np.array
            y_hat - predicted classes for X
        """
        # scale the data
        X_scaled = self.scaler1.transform(X)
        # get PCA and select n features
        X_pca = self.pca.transform(X_scaled)
        X_pca = X_pca[:, :self.n_pca]
        X_pca_scaled = self.scaler2.transform(X_pca)
        # predict
        y_hat = self.rfc.predict(X_pca_scaled)
        return y_hat

    # Evaluate
    def evaluate(self, y: np.array, y_hat: np.array):
        cm = confusion_matrix(y, y_hat)

        cm_pd = pd.DataFrame(cm, index=['actual FAIL', 'actual PASS'],
                             columns=['predicted FAIL', 'predicted PASS'])
        TP = cm_pd.loc['actual FAIL', 'predicted FAIL']
        FP = cm_pd.loc['actual PASS', 'predicted FAIL']
        FN = cm_pd.loc['actual FAIL', 'predicted PASS']
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        return cm_pd, precision, recall, f1
