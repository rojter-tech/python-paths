#%%
import os
thispath = '/home/dreuter/Github/python-paths/Python_A-Z'
os.chdir(thispath)
print(os.getcwd())
#%% #Categorical varibles
import pandas as pd
df = pd.read_csv("P4-Movie-Ratings.csv")
#%%
df.head()
#%%
