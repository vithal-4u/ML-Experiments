#Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datasets = pd.read_csv('Salary_Data.csv')
print(datasets)

#taking independent and depended variables from dataset and convert into matrixs
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 1].values

#Splitting dataset into train and test.
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predict the Test set result
Y_pred = regressor.predict(X_test)

#Visualising the training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train))
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the testing set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train))
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()