# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:28:57 2022

@author: dshre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sklearn.cluster as cluster
# err_ranges.py needs to be in the same folder as this python file.
from err_ranges import err_ranges

def growthfunct(t, s, k):
    '''
    function to get fitted values of exponential
    population growth.

    Parameters
    ----------
    t : int
        Year
    s : int
        population
    k : float
        Growth rate

    Returns
    -------
    op : TYPE
        DESCRIPTION.

    '''
    op = s * np.exp(k * (t - 2008))
    
    return op

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


def kmeans(df,x,y):
    '''
    The function plots KMeans Clusters.

    Parameters
    ----------
    df : Dataframe
        Dataframe from which data is to be taken.
    x : String
        1st column to be considered for clustering.
    y : String
        2nd column to be considered for clustering.

    Returns
    -------
    None.

    '''
    xlab = x
    ylab = y
    x = df[x]
    y = df[y]

    kmeans = cluster.KMeans(n_clusters=4)
    kmeans.fit(df)

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
    plt.xlabel(xlab)
    plt.ylabel(ylab)    

# Reading the dataframe  
df_cc, df_ccT = read_data("Dataset.csv")

# Making seperate dataframe for data fitting
df_ind = df_cc[df_cc["Country Name"] == "India"]
print(df_cc.describe())
print(df_cc.corr())
print(df_cc.head())

# Normalising the data for clustering.
for col in df_cc.columns[2:]:
    df_cc[col] = norm(df_cc[col])

# Clustering features against total population.
df_CO_TP = df_cc[["CO2 emissions (metric tons per capita)", 
                  "Population total"]].copy()
kmeans(df_CO_TP,"CO2 emissions (metric tons per capita)","Population total")

df_UP_TP = df_cc[["Urban population",
                  "Population total"]].copy()
kmeans(df_UP_TP,"Urban population","Population total")

df_PG_TP = df_cc[["Population growth (annual %)",
                  "Population total"]].copy()
kmeans(df_PG_TP,"Population growth (annual %)","Population total")


param, pcovar = opt.curve_fit(growthfunct, df_ind["Time"], 
                              df_ind["Population total"])

sigma = np.sqrt(np.diag(pcovar))
low, up = err_ranges(df_ind["Time"], growthfunct, param, sigma)

print(param, pcovar, sep="\n")

# Plotting the population growth and fitted population growth.
df_ind["pop_exp"] = growthfunct(df_ind["Time"],*param)

plt.figure()
plt.plot(df_ind["Time"], df_ind["Population total"],
         label="data", color = "Red")
plt.plot(df_ind["Time"], df_ind["pop_exp"], 
         label="fit", color = "Black")
plt.fill_between(df_ind["Time"], low, up, 
                 label="fill", alpha=0.7, color = "Yellow")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Population Growth in India")
plt.legend()

print("Forcasted population")
low30, up30 = err_ranges(2030, growthfunct, param, sigma)
print("2030 between ", low30, "and", up30)

print("Forcasted population")
low40, up40 = err_ranges(2040, growthfunct, param, sigma)
print("2040 between ", low40, "and", up40)