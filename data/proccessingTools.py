# Libries 
# ----------- to wortk with arrays (numphy)
# ----------- to plot charts (matplotlip) and graphs
# ----------- to import data set (pandas)
# ----------- for missing data sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pandas as pd

# ----------- importing data set ---------------
dataset = pd.read_csv('Data.csv')

# iloc (locate indexes (to get the data of our constant variables[excepct the last colum]))
# get all the rows of the data set (:)
# get the colums execept the last one (:-1)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# --------- taking care of missing data -----------
# --- How ----
# 1. by deleting the cells of the missing data
# 2. replace the missing value with the average of all the cell (data) value
# --- Tools ---
# using tool called sklearn.impute

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

# ----- encoding categorical data -----
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

