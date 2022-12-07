# Databricks notebook source
print("Running the test2.py script")

# COMMAND ----------

import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow


mlflow.set_experiment("/Users/shanmukha.garime@tigeranalytics.com/demo/iris_experiment")

iris = sns.load_dataset('iris')
iris.head()

# COMMAND ----------

def train_test_dataset(x,y):
    x_train,x_test ,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    return x_train,x_test ,y_train,y_test

# COMMAND ----------

def metrics(predicted,actual):
    f1 = f1_score(actual,predicted , average='macro')
    precision = precision_score(actual,predicted , average='macro')
    recall = recall_score(actual,predicted , average='macro')

    return f1,precision,recall

# COMMAND ----------

x = iris.drop("species",axis=1)
y = iris[["species"]]

standard_model = StandardScaler()
standard_model.fit(x)
x_trained = standard_model.transform(x)

with mlflow.start_run() as deom_run:
    x_train,x_test,y_train,y_test = train_test_dataset(x_trained,y) 

    model = LogisticRegression()
    model = model.fit(x_train,y_train)
    predicted = model.predict(x_test)

    f1,precision,recall = metrics(y_test , predicted)
    
    mlflow.log_metric("f1_score" , f1)
    mlflow.log_metric("precision_score" , precision)
    mlflow.log_metric("Recall Score" , recall)
    
    mlflow.sklearn.log_model(model,"logistic Regression model")

# COMMAND ----------


