#%%
import os
thispath = '/home/dreuter/Github/python-paths/Python_A-Z'
os.chdir(thispath)
print(os.getcwd())
#%% #Categorical varibles
import pandas as pd
movies = pd.read_csv("P4-Movie-Ratings.csv")
movies.info()
#%%
movies.columns = ['Film', 'Genre', 'CriticRating', 'AudienceRating',
              'BudgetMillion', 'Year']
movies.Film = movies.Film.astype('category')
movies.Genre = movies.Genre.astype('category')
movies.Year = movies.Year.astype('category')
#%%
movies.info()
#%%
movies.head()
#%%
movies.describe()
#%%
movies.Genre.cat.categories
#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#%%
#Jointplots
j1 = sns.jointplot(data=movies, x='CriticRating', y='AudienceRating', kind='hex')
#%%
# Histograms
m1 = sns.distplot(movies.AudienceRating, bins=15)
#%%
m2 = sns.distplot(movies.CriticRating, bins=15)
#%%
n1 = plt.hist(movies.AudienceRating, bins=15)
#%%
n1 = plt.hist(movies.CriticRating, bins=15)
#%%
plt.hist(movies[movies.Genre == 'Drama'].BudgetMillion, bins=15)
plt.hist(movies[movies.Genre == 'Action'].BudgetMillion, bins=15)
plt.hist(movies[movies.Genre == 'Thriller'].BudgetMillion, bins=15)
plt.show()
#%%
#Stacked Histograms
stackedlist = [movies[movies.Genre == 'Drama'].BudgetMillion,
movies[movies.Genre == 'Action'].BudgetMillion,
movies[movies.Genre == 'Thriller'].BudgetMillion]
plt.hist(stackedlist, bins=15, stacked=True, label=['Drama','Action','Thriller'])
plt.legend()
plt.show()
#%%
genres = list(movies.Genre.cat.categories)
stackedlist = []
for genre in genres:
       stackedlist.append(movies[movies.Genre == genre].BudgetMillion)
plt.hist(stackedlist, bins=30, stacked=True, label=genres)
plt.legend()
plt.show()
#%%
# KDE Plot
vis1 = sns.lmplot(data=movies,
                     x='CriticRating',
                     y='AudienceRating',
                     fit_reg=False,
                     hue='Genre',
                     height=7,
                     aspect=1)

#%%
k1 = sns.kdeplot(movies.CriticRating,
                 movies.AudienceRating,
                 shade=True,
                 shade_lowest=False,
                 cmap='Reds')
k1b = sns.kdeplot(movies.CriticRating,
                  movies.AudienceRating,
                  cmap='Reds')
plt.show()
#%%
k1 = sns.kdeplot(movies.BudgetMillion,
                 movies.CriticRating)

#%%
# Subplots
f, ax = plt.subplots(1,2)

#%%
