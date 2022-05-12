# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:28:57 2022

@author: dshre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster

def read_data(fname):
    """
    The function reads data from a csv file
    and returns a Pandas Dataframe.

    Parameters
    ----------
    fname : string
        Name of file containing data of a World Bank indicator.

    Returns
    -------
    df : Pandas Dataframe
        A Pandas Dataframe containing relevent data
        from the World Bank indicator file i.e. Countries X Years data.
    df.T : Pandas Dataframe
        Transpose of the df Dataframe where the Years are the Index.

    """
    df = pd.read_csv(fname)
    
    return df, df.T

def norm(column):
    '''
    The function takes 1 column of a dataframe
    and returns the normalised values for the column.

    Parameters
    ----------
    column : Series
        input is a series of a column of the dataframe.

    Returns
    -------
    scaled_vals : Series
        the scaled/ normalised values for the input column are returned.

    '''
    col_min =  np.min(column)
    col_max = np.max(column)
    scaled_vals = (column - col_min) / (col_max - col_min)
    
    return scaled_vals

def makeplot(df, col1, col2):
    '''
    function takes a dataframe and 2 column names as input
    and makes a scatter plot for the 2 columns.

    Parameters
    ----------
    df : DataFrame
        variable for the dataframe.
    col1 : String
        name of 1st column.
    col2 : String
        name of 2nd column.

    Returns
    -------
    None.

    '''
    plt.figure()
    plt.scatter(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()
    
    
df_cc, fd_ccT = read_data("Dataset.csv")
print(df_cc.describe())
print(df_cc.corr())
print(df_cc.head())

for col in df_cc.columns[2:]:
    df_cc[col] = norm(df_cc[col])


df_CO_TP = df_cc[["CO2 emissions (metric tons per capita)", "Population total"]].copy()

#KMeans Clusteing
x = df_CO_TP["CO2 emissions (metric tons per capita)"]
y = df_CO_TP["Population total"]

kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(df_CO_TP)

labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure()
col = ["red", "green", "blue", "yellow"]
for l in range (0,4):
    plt.plot(x[labels==l], y[labels==l], "o", markersize=3, color=col[l])

for ic in range(4):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.title("KMeans Clustering")