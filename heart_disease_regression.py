"""
Multiple Linear Regression uses several explanatory variables to predict the outcome of a response variable. 

Dataset link: https://github.com/bnsreenu/python_for_microscopists/tree/master/268-How%20to%20deploy%20your%20trained%20machine%20learning%20model%20into%20a%20local%20web%20application/files_for_training_model

#Heart disease: The effect that the independent variables biking and smoking have on the dependent variable heart disease.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

# Importing the dataset
df = pd.read_csv('heart_data.csv')
print(df.head())