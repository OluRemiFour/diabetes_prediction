import numpy as py
import pandas as pd
import sklearn.datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


# importing dataset from sklearn
house_price_dataset = sklearn.datasets.load_boston()

# loading data into pandas Data frame 
house_price_data_frame = pd.DataFrame(house_price_dataset.data)
print(house_price_data_frame)