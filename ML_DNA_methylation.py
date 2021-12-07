#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


cwd = os.getcwd()

os.chdir(r"C:\Users\NIRER\Desktop\Python\Advanced")
methylation= pd.read_csv("methylation_data_merged_all.csv")
methylation.head()
methylation=methylation.dropna()
X=methylation.drop(columns=['Accept.','status'])
y=methylation['status']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


