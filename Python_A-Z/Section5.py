#%%
import pandas as pd
stats = pd.read_csv("~/Github/python-paths/Python_A-Z/P4-Demographic-Data.csv")
stats.head()

#%%
import os
print(os.getcwd())

#%%
len(stats) #195 rows imported

#%%
type(stats)

#%%
stats.columns

#%%
len(stats.columns) # 5 number of cols

#%%
stats.head()

#%%
stats.tail()

#%%
# information on the columns
stats.info()

#%%
# stats on the columns
stats.describe()

#%%
stats.describe().transpose()

#%%
stats.head()

#%%
stats.columns

#%%
stats.columns = ['CountryName', 'CountryCode', 'BirthRate', 'InternetUsers',
       'IncomeGroup']

#%%
stats.head()

#%%
stats[21:26] # discrimination regards rows

#%%
stats[:10]

#%%
stats[-1:0:-1].head()

#%%
stats[::20]

#%%
stats.columns

#%%
stats[['CountryName', 'BirthRate']].head()

#%%
stats[['BirthRate', 'CountryName']].head()

#%%
stats.head()

#%%
stats.BirthRate #relies on keeping columns name simple one-word labels

#%%
stats[4:8][['CountryName', 'BirthRate']]

#%%
stats[['CountryName', 'BirthRate']][4:8]

#%%
stats.head()

#%%
stats[['CountryCode','BirthRate','InternetUsers']]

#%%
result = stats.BirthRate * stats.InternetUsers # element by element
print(result)

#%%
stats['MyCalc'] = result

#%%
stats.head()

#%%
stats.drop('MyCalc', 1).head()

#%%
stats.head()

#%%
stats = stats.drop('MyCalc', 1)

#%%
stats.head()

#%%
## Filtering is abount rows
stats.head()

#%%
stats.InternetUsers < 2

#%%
Filter = stats.InternetUsers < 2

#%%
stats[Filter]

#%%
stats[stats['BirthRate'] > 40]
stats[stats.BirthRate > 40]

#%%
stats[(stats.BirthRate > 40) and (stats.InternetUsers < 2)] # error, exprect singe values

#%%
Filter1 = stats.InternetUsers < 2
Filter2 = stats.BirthRate > 40

#%%
Filter1 & Filter2

#%%
stats[Filter1 & Filter2] # umpersand operator &, elementwise

#%%
stats.head()

#%%
stats[stats.IncomeGroup == 'High income']

#%%
stats.IncomeGroup.unique()

#%%
stats[stats.CountryName == 'Malta']

#%%
stats[stats.InternetUsers > 94.78]

#%%
stats[2:3]['CountryName'] # returns series

#%%
stats.iat[3,4] # returns object type, string
#%%
stats.iat[0,3] # returns specififc object type, float64
#%%
stats.at[2,'BirthRate'] # label columns instead of index

#%%
sub10 = stats[::10]
sub10
#%%
sub10.iat[10,0] # counts index

#%%
sub10.at[10,'CountryName'] # treats index as labels

#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = 10,4

#%%
vis1 = sns.distplot(stats["InternetUsers"])

#%%
vis1 = sns.distplot(stats["InternetUsers"], bins=30)

#%%
vis2 = sns.boxplot(data=stats, x='IncomeGroup', y='BirthRate')

#%%
vis3 = sns.lmplot(data=stats, x='InternetUsers', y='BirthRate')

#%%
vis3 = sns.lmplot(data=stats, x='InternetUsers', y='BirthRate', \
                  fit_reg=False, hue='IncomeGroup', size=10,    \
                  scatter_kws={"s":150})
#%%
