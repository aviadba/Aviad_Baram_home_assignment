"""
This module implements report visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def showRawDataDtype(features: pd.DataFrame):
    """
    Generate a bar plot of the data types of the raw data
    :param features: pd.DataFrome
        Raw data fraqme
    """
    # Get the feature column dtypes and their counts
    dtype_counts = features.dtypes.value_counts()
    # Plotting
    plt.figure(figsize=(8, 6))
    dtype_counts.plot(kind='bar', color='skyblue')
    # Add labels and title
    plt.title('Data Types of features columns', fontsize=14)
    plt.xlabel('Data Type', fontsize=12)
    plt.ylabel('Count of Columns', fontsize=12)
    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def showImbalance(response: pd.Series):
    """
    Generate a bar plot of imbalance between test pass and test fail
    :param response:
    :return:
    """
    response_counts = response.value_counts()
    # Plotting
    plt.figure(figsize=(8, 6))
    response_counts.plot(kind='bar', color='skyblue')
    # Add labels and title
    plt.title('Test PASS/FAIL', fontsize=14)
    plt.xlabel('Test FAIL', fontsize=12)
    plt.ylabel('Count of Samples', fontsize=12)
    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def showVarVSPCA(var_cmsum: np.array, n_pca: int):
    """
    Generate a plot of cumulative explain variance vs the number of PCA componenets
    :param var_cmsum: np.array
        Cumulative sum of explained variance by PCA model
    :param n_pca: int
        The number of pca parameters that are selected
    :return:
    """
    plt.plot(var_cmsum)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axvline(linewidth=2, color='r', linestyle='--', x=n_pca, ymin=0, ymax=1)
    plt.show()

def plotPCAvaraince(var_cumsum: np.array, pca_num:int):
    """
    Generate a plot of explained variance vs num of PCA components
    :param var_cumsum: np.array
        cummulative sum of the explained variance by PCA
    :param pca_num: int
        Number of PCA components selected
    """
    plt.plot(var_cumsum)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axvline(linewidth=3, color='r', linestyle='--', x=pca_num, ymin=0, ymax=1)
    plt.show()
