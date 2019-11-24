#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#loading datasets
datasets = pd.read_csv("50_Startups.csv")

#taking independent and depended variables from dataset and convert into matrixs
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 4].values

#Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:, 3])

#Convert the lable into OneHotEncoder to make no lable is bigger or compared.
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# Avoiding Dummy Variable trap
X = X[:, 1:]


#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predict the Test set result
Y_pred = regressor.predict(X_test)

# Building optimal model using Backward elimination
#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50,1)).astype(int), values= X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS._results.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS._results.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS._results.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS._results.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS._results.summary()

# Doing Plot on 1 independent Variable Done by my one
X_opt_1 = X_opt[:,1]

regressor1 = LinearRegression()
regressor1.fit(X_opt_1.reshape(-1, 1),Y.reshape(-1, 1))
 
#Predict the Test set result
Y_pred1 = regressor1.predict(X_test[:,2].reshape(-1, 1))

#Visualising the training set results
plt.scatter(X_opt_1.reshape(-1, 1), Y.reshape(-1, 1), color='red')
plt.plot(X_opt_1.reshape(-1, 1), regressor1.predict(X_opt_1.reshape(-1, 1)))
plt.title('R&D Spend vs Profit (Training set)')
plt.xlabel('R & D Spend')
plt.ylabel('Profit')
plt.show()

