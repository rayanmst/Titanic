# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:11:42 2019

@author: Rayan Martins Steinbach
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_map( df ):
    corr = passengersData.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(corr, cmap = cmap, square=True, 
                    cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, 
                    annot_kws = { 'fontsize' : 12 })

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

path = r'train.csv'
passengersData = pd.read_csv(path, index_col = 'PassengerId')

# Exploratory data Analysis:

print(passengersData['Embarked'].describe())
print(passengersData['Embarked'].unique(), end = '\n\n')

print(passengersData['Sex'].describe())
print(passengersData['Sex'].unique(), end = '\n\n')

# Completed the missing lines with 'S'
passengersData['Embarked'].fillna('S', inplace = True)
plot_correlation_map(passengersData)

# Grouping some columns for better analysis
classSurvivors = passengersData[['Pclass', 'Survived']].groupby(['Pclass'], 
      as_index=False).mean().sort_values(by='Survived', ascending=False)

#Creating some graphs for a better visualization
cs = sns.barplot(y = classSurvivors['Survived'], x = classSurvivors['Pclass'])

#plot_distribution(passengersData, var = 'Age', target = 'Survived', row = 'Sex')

g = sns.FacetGrid(passengersData, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(passengersData, col = 'Survived', row = 'Pclass',
                     height = 2.2, aspect = 1.6)
grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
grid.add_legend();

grid = sns.FacetGrid(passengersData, row = 'Embarked', height = 2.2,
                     aspect = 1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

grid = sns.FacetGrid(passengersData, row = 'Embarked', col = 'Survived', 
                     height = 2.2, aspect = 1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = None)

#Changing Strings elements for integer
passengersData['Sex'] = passengersData['Sex'].map({'female': 0, 'male': 1})
print(passengersData.info(), end = '\n\n')
