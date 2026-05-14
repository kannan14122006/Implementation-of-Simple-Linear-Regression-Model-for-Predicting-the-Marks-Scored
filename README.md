# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (pandas, numpy, matplotlib, and sklearn) for data handling, model training, visualization, and performance evaluation.

2.Load the dataset (data.csv) into a DataFrame and inspect the data using .head() and .tail() to understand its structure.

3.Separate the dataset into input features X (independent variable) and output y (dependent variable) for model training.

4.Split the data into training and testing sets using train_test_split() to evaluate model performance on unseen data.

5.Initialize and train a LinearRegression model on the training data using the .fit() method.

6.Predict the output for the test set using .predict() and compare predicted values (y_pred) with actual test values (y_test).

7.Visualize the training and test results using scatter plots and regression lines to assess the fit of the model visually.

8.Evaluate model accuracy using error metrics like MAE, MSE, and RMSE, which quantify prediction errors in different ways.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KANNAN R
RegisterNumber:212224240072
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:


Head values


<img width="167" height="241" alt="image" src="https://github.com/user-attachments/assets/946e1c48-f77e-4895-8bcb-62aea9664fb0" />



Trail values


<img width="178" height="235" alt="image" src="https://github.com/user-attachments/assets/efdeb5fb-d757-4731-8aa8-e23bae8c204a" />

x Values


<img width="718" height="60" alt="image" src="https://github.com/user-attachments/assets/3b9e6f15-52d5-4f01-8860-3d4370e6b84d" />



y Values


Predicted values

<img width="698" height="74" alt="image" src="https://github.com/user-attachments/assets/a906585e-0bdc-4dee-a8d8-1cd6e98b1d17" />



Actual values

<img width="576" height="28" alt="image" src="https://github.com/user-attachments/assets/82f7c4bd-a2ec-4bdf-920d-1011296639af" />



Training set


<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/a541046a-1f0d-49d2-b869-d426c5b6cd16" />



Testing set


<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/349245ff-3a66-4141-ab77-8f43a556d470" />



MSE, MAE and RMSE


<img width="258" height="66" alt="image" src="https://github.com/user-attachments/assets/f1e9a4a5-9a2a-4314-8184-f65475b2eecc" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
