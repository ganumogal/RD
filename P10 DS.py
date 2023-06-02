#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:58:02 2022

@author: cg
"""
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv("iris.csv", header=None)
col_names = ['sepal_length','sepal_width','petal_length','petal_width','species']
df = pd.read_csv("iris.csv", names=col_names)
print(df.head())
column = len(list(df))
print(column)
print(df.info())
print(np.unique(df["species"]))

fig, axes = plt.subplots(2,2,figsize=(16,8))

axes[0,0].set_title("Distribution for first column")
axes[0,0].hist(df["sepal_length"])

axes[0,1].set_title("Distribution for second column")
axes[0,1].hist(df["sepal_width"])

axes[1,0].set_title("Distribution for thirs column")
axes[1,0].hist(df["petal_length"])

axes[1,1].set_title("Distribution for fourth column")
axes[1,1].hist(df["petal_width"])

data_to_plot = [df["sepal_length"], df["sepal_width"], df["petal_length"], df["petal_width"]]

sns.set_style("whitegrid")
fig = plt.figure(1, figsize =(12,8))

ax=fig.add_subplot(111)

bp=ax.boxplot(data_to_plot)

print(df.describe())
