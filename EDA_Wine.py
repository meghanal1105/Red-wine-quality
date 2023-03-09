# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 18:40:03 2021

@author: 91876
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import ols
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import svm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

wine= pd.read_csv("C:/Users/91876/Desktop/Kaggle Individual/4_Red Wine Quality/winequality-red.csv")
df= pd.DataFrame(wine)
df.isna().sum()
df.describe()
df.info()
df.columns
df.shape

''' Target variable '''
''' quality '''
df.quality.isna().sum()
df.quality.value_counts().sort_index()
df.quality.describe()
df.quality.unique()
''' Categorical '''
# No null values

#Countplot
sns.countplot(x ='quality', data = df)

df['quality'].replace(3,1,inplace = True) # Replacing 3 by 1
df['quality'].replace(4,2,inplace = True) # Replacing 4 by 2
df['quality'].replace(5,3,inplace = True) # Replacing 5 by 3
df['quality'].replace(6,4,inplace = True) # Replacing 6 by 4
df['quality'].replace(7,5,inplace = True) # Replacing 7 by 5
df['quality'].replace(8,6,inplace = True) # Replacing 8 by 6
df.quality.value_counts().sort_index()

#Countplot after replacing
sns.countplot(x ='quality', data = df)




df.info()
'''###########  1- fixed acidity  ################'''
df= df.rename(columns= {'fixed acidity':'fixed_acidity'})
df.info()
df.fixed_acidity.isna().sum()
df.fixed_acidity.value_counts().sort_index()
df.fixed_acidity.describe()
df.fixed_acidity.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.fixed_acidity, color = 'red')
plt.xlabel('fixed_acidity')
plt.title('Histogram of fixed_acidity')

#Boxplot
plt.boxplot(df['fixed_acidity'],1,'rs',1)
plt.xlabel('fixed_acidity')
plt.ylabel('counts')
plt.title('Boxplot of fixed_acidity')

# there are outliers

# Outliers Count
IQR1 = df['fixed_acidity'].quantile(0.75) - df['fixed_acidity'].quantile(0.25)
IQR1

UL1 = df['fixed_acidity'].quantile(0.75) + (1.5*IQR1)
UL1

df.fixed_acidity[(df.fixed_acidity > UL1)].value_counts().sum()
# 49

df.fixed_acidity = np.where(df.fixed_acidity > UL1, UL1, df.fixed_acidity)

df.fixed_acidity[(df.fixed_acidity > UL1)].value_counts().sum()
# 0

#Boxplot after outliers treatment
plt.boxplot(df['fixed_acidity'],1,'rs',1)
plt.xlabel('fixed_acidity')
plt.ylabel('counts')
plt.title('Boxplot of fixed_acidity')




'''###########  2- volatile acidity  ###############'''
df= df.rename(columns={'volatile acidity':'volatile_acidity'})
df.info()
df.volatile_acidity.isna().sum()
df.volatile_acidity.value_counts().sort_index()
df.volatile_acidity.describe()
df.volatile_acidity.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.volatile_acidity, color = 'red')
plt.xlabel('volatile_acidity')
plt.title('Histogram of volatile_acidity')

#Boxplot
plt.boxplot(df['volatile_acidity'],1,'rs',1)
plt.xlabel('volatile_acidity')
plt.ylabel('counts')
plt.title('Boxplot of volatile_acidity')

# there are outliers

# Outliers Count
IQR2 = df['volatile_acidity'].quantile(0.75) - df['volatile_acidity'].quantile(0.25)
IQR2

UL2 = df['volatile_acidity'].quantile(0.75) + (1.5*IQR2)
UL2

df.volatile_acidity[(df.volatile_acidity > UL2)].value_counts().sum()
# 19

df.volatile_acidity = np.where(df.volatile_acidity > UL2, UL2, df.volatile_acidity)

df.volatile_acidity[(df.volatile_acidity > UL2)].value_counts().sum()
# 0

#Boxplot after outliers treatment
plt.boxplot(df['volatile_acidity'],1,'rs',1)
plt.xlabel('volatile_acidity')
plt.ylabel('counts')
plt.title('Boxplot of volatile_acidity')





'''############  3- citric acid  ##############'''
df= df.rename(columns={'citric acid':'citric_acid'})
df.info()
df.citric_acid.isna().sum()
df.citric_acid.value_counts().sort_index()
df.citric_acid.describe()
df.citric_acid.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.citric_acid, color = 'red')
plt.xlabel('citric_acid')
plt.title('Histogram of citric_acid')

#Boxplot
plt.boxplot(df['citric_acid'],1,'rs',1)
plt.xlabel('citric_acid')
plt.ylabel('counts')
plt.title('Boxplot of citric_acid')

# there are outliers

# Outliers Count
IQR3 = df['citric_acid'].quantile(0.75) - df['citric_acid'].quantile(0.25)
IQR3

UL3 = df['citric_acid'].quantile(0.75) + (1.5*IQR3)
UL3

df.citric_acid[(df.citric_acid > UL3)].value_counts().sum()
# 1

df.citric_acid = np.where(df.citric_acid > UL3, UL3, df.citric_acid)

df.citric_acid[(df.citric_acid > UL3)].value_counts().sum()
# 0

#Boxplot after outliers treatment
plt.boxplot(df['citric_acid'],1,'rs',1)
plt.xlabel('citric_acid')
plt.ylabel('counts')
plt.title('Boxplot of citric_acid')





'''############  4- residual sugar  #############'''
df= df.rename(columns={'residual sugar':'residual_sugar'})
df.info()
df.residual_sugar.isna().sum()
df.residual_sugar.value_counts().sort_index()
df.residual_sugar.describe()
df.residual_sugar.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.residual_sugar, color = 'red')
plt.xlabel('residual_sugar')
plt.title('Histogram of residual_sugar')

#Boxplot
plt.boxplot(df['residual_sugar'],1,'rs',1)
plt.xlabel('residual_sugar')
plt.ylabel('counts')
plt.title('Boxplot of residual_sugar')

# there are outliers

# Outliers Count
IQR4 = df['residual_sugar'].quantile(0.75) - df['residual_sugar'].quantile(0.25)
IQR4

UL4 = df['residual_sugar'].quantile(0.75) + (1.5*IQR4)
UL4

df.residual_sugar[(df.residual_sugar > UL4)].value_counts().sum()
# 155

df.residual_sugar = np.where(df.residual_sugar > UL4, UL4, df.residual_sugar)

df.residual_sugar[(df.residual_sugar > UL4)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['residual_sugar'],1,'rs',1)
plt.xlabel('residual_sugar')
plt.ylabel('counts')
plt.title('Boxplot of residual_sugar')






'''###########  5- chlorides ##############'''
df.chlorides.isna().sum()
df.chlorides.value_counts().sort_index()
df.chlorides.describe()
df.chlorides.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.chlorides, color = 'red')
plt.xlabel('chlorides')
plt.title('Histogram of chlorides')

#Boxplot
plt.boxplot(df['chlorides'],1,'rs',1)
plt.xlabel('chlorides')
plt.ylabel('counts')
plt.title('Boxplot of chlorides')

# there are outliers

# Outliers Count
IQR5 = df['chlorides'].quantile(0.75) - df['chlorides'].quantile(0.25)
IQR5

UL5 = df['chlorides'].quantile(0.75) + (1.5*IQR5)
UL5

LL5 = df['chlorides'].quantile(0.25) - (1.5*IQR5)
LL5

df.chlorides[(df.chlorides > UL5)].value_counts().sum()
# 103
df.chlorides[(df.chlorides < LL5)].value_counts().sum()
# 9
df.chlorides = np.where(df.chlorides > UL5, UL5, df.chlorides)

df.chlorides = np.where(df.chlorides < LL5, LL5, df.chlorides)

df.chlorides[(df.chlorides > UL5)].value_counts().sum()
# 0
df.chlorides[(df.chlorides < LL5)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['chlorides'],1,'rs',1)
plt.xlabel('chlorides')
plt.ylabel('counts')
plt.title('Boxplot of chlorides')





'''###########  6- free sulfur dioxide #############'''
df= df.rename(columns={'free sulfur dioxide':'free_sulfur_dioxide'})
df.info()
df.free_sulfur_dioxide.isna().sum()
df.free_sulfur_dioxide.value_counts().sort_index()
df.free_sulfur_dioxide.describe()
df.free_sulfur_dioxide.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.free_sulfur_dioxide, color = 'red')
plt.xlabel('free_sulfur_dioxide')
plt.title('Histogram of free_sulfur_dioxide')

#Boxplot
plt.boxplot(df['free_sulfur_dioxide'],1,'rs',1)
plt.xlabel('free_sulfur_dioxide')
plt.ylabel('counts')
plt.title('Boxplot of free_sulfur_dioxide')

# there are outliers

# Outliers Count
IQR6 = df['free_sulfur_dioxide'].quantile(0.75) - df['free_sulfur_dioxide'].quantile(0.25)
IQR6

UL6 = df['free_sulfur_dioxide'].quantile(0.75) + (1.5*IQR6)
UL6

df.free_sulfur_dioxide[(df.free_sulfur_dioxide > UL6)].value_counts().sum()
# 30
df.free_sulfur_dioxide = np.where(df.free_sulfur_dioxide > UL6, UL6, df.free_sulfur_dioxide)

df.free_sulfur_dioxide[(df.free_sulfur_dioxide > UL6)].value_counts().sum()
# 0





'''#############  7- total sulfur dioxide  ###############'''
df= df.rename(columns={'total sulfur dioxide':'total_sulfur_dioxide'})
df.info()
df.total_sulfur_dioxide.isna().sum()
df.total_sulfur_dioxide.value_counts().sort_index()
df.total_sulfur_dioxide.describe()
df.total_sulfur_dioxide.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.total_sulfur_dioxide, color = 'red')
plt.xlabel('total_sulfur_dioxide')
plt.title('Histogram of total_sulfur_dioxide')

#Boxplot
plt.boxplot(df['total_sulfur_dioxide'],1,'rs',1)
plt.xlabel('total_sulfur_dioxide')
plt.ylabel('counts')
plt.title('Boxplot of total_sulfur_dioxide')

# there are outliers

# Outliers Count
IQR7 = df['total_sulfur_dioxide'].quantile(0.75) - df['total_sulfur_dioxide'].quantile(0.25)
IQR7

UL7 = df['total_sulfur_dioxide'].quantile(0.75) + (1.5*IQR7)
UL7

df.total_sulfur_dioxide[(df.total_sulfur_dioxide > UL7)].value_counts().sum()
# 55

df.total_sulfur_dioxide = np.where(df.total_sulfur_dioxide > UL7, UL7, df.total_sulfur_dioxide)

df.total_sulfur_dioxide[(df.total_sulfur_dioxide > UL7)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['total_sulfur_dioxide'],1,'rs',1)
plt.xlabel('total_sulfur_dioxide')
plt.ylabel('counts')
plt.title('Boxplot of total_sulfur_dioxide')





'''#############  8- density  #################'''
df.density.isna().sum()
df.density.value_counts().sort_index()
df.density.describe()
df.density.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.density, color = 'red')
plt.xlabel('density')
plt.title('Histogram of density')

#Boxplot
plt.boxplot(df['density'],1,'rs',1)
plt.xlabel('density')
plt.ylabel('counts')
plt.title('Boxplot of density')

# there are outliers

# Outliers Count
IQR8 = df['density'].quantile(0.75) - df['density'].quantile(0.25)
IQR8

UL8 = df['density'].quantile(0.75) + (1.5*IQR8)
UL8

LL8 = df['density'].quantile(0.25) - (1.5*IQR8)
LL8

df.density[(df.density > UL8)].value_counts().sum()
# 24
df.density[(df.density < LL8)].value_counts().sum()
# 21

df.density = np.where(df.density > UL8, UL8, df.density)

df.density = np.where(df.density < LL8, LL8, df.density)

df.density[(df.density > UL8)].value_counts().sum()
# 0
df.density[(df.density < LL8)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['density'],1,'rs',1)
plt.xlabel('density')
plt.ylabel('counts')
plt.title('Boxplot of density')





'''#############  9- pH  ################'''
df.pH.isna().sum()
df.pH.value_counts().sort_index()
df.pH.describe()
df.pH.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.pH, color = 'red')
plt.xlabel('pH')
plt.title('Histogram of pH')

#Boxplot
plt.boxplot(df['pH'],1,'rs',1)
plt.xlabel('pH')
plt.ylabel('counts')
plt.title('Boxplot of pH')

# there are outliers

# Outliers Count
IQR9 = df['pH'].quantile(0.75) - df['pH'].quantile(0.25)
IQR9

UL9 = df['pH'].quantile(0.75) + (1.5*IQR9)
UL9

LL9 = df['pH'].quantile(0.25) - (1.5*IQR9)
LL9

df.pH[(df.pH > UL9)].value_counts().sum()
# 21
df.pH[(df.pH < LL9)].value_counts().sum()
# 14

df.pH = np.where(df.pH > UL9, UL9, df.pH)

df.pH = np.where(df.pH < LL9, LL9, df.pH)

df.pH[(df.pH > UL9)].value_counts().sum()
# 0
df.pH[(df.pH < LL9)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['pH'],1,'rs',1)
plt.xlabel('pH')
plt.ylabel('counts')
plt.title('Boxplot of pH')




'''############  10- sulphates  #############'''
df.sulphates.isna().sum()
df.sulphates.value_counts().sort_index()
df.sulphates.describe()
df.sulphates.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.sulphates, color = 'red')
plt.xlabel('sulphates')
plt.title('Histogram of sulphates')

#Boxplot
plt.boxplot(df['sulphates'],1,'rs',1)
plt.xlabel('sulphates')
plt.ylabel('counts')
plt.title('Boxplot of sulphates')

# there are outliers

# Outliers Count
IQR10 = df['sulphates'].quantile(0.75) - df['sulphates'].quantile(0.25)
IQR10

UL10 = df['sulphates'].quantile(0.75) + (1.5*IQR10)
UL10

df.sulphates[(df.sulphates > UL10)].value_counts().sum()
# 59

df.sulphates = np.where(df.sulphates > UL10, UL10, df.sulphates)

df.sulphates[(df.sulphates > UL10)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['sulphates'],1,'rs',1)
plt.xlabel('sulphates')
plt.ylabel('counts')
plt.title('Boxplot of sulphates')





'''############  11- alcohol  ##############'''
df.alcohol.isna().sum()
df.alcohol.value_counts().sort_index()
df.alcohol.describe()
df.alcohol.unique()
# no null values
''' Continuous '''

#Histogram
sns.distplot(df.alcohol, color = 'red')
plt.xlabel('alcohol')
plt.title('Histogram of alcohol')

#Boxplot
plt.boxplot(df['alcohol'],1,'rs',1)
plt.xlabel('alcohol')
plt.ylabel('counts')
plt.title('Boxplot of alcohol')

# there are outliers

# Outliers Count
IQR11 = df['alcohol'].quantile(0.75) - df['alcohol'].quantile(0.25)
IQR11

UL11 = df['alcohol'].quantile(0.75) + (1.5*IQR11)
UL11

df.alcohol[(df.alcohol > UL11)].value_counts().sum()
# 13

df.alcohol = np.where(df.alcohol > UL11, UL11, df.alcohol)

df.alcohol[(df.alcohol > UL11)].value_counts().sum()
# 0

#Boxplot after treatment
plt.boxplot(df['alcohol'],1,'rs',1)
plt.xlabel('alcohol')
plt.ylabel('counts')
plt.title('Boxplot of alcohol')



''' EDA is done '''


df.to_csv("C:/Users/91876/Desktop/Kaggle Individual/5_Red Wine Quality/Exported files/EDA_RedWine.csv")
