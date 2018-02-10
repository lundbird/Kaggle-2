import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

test = pd.read_csv(r'C:\Users\alex\workspace\housingprices\test.csv',index_col='Id')
train = pd.read_csv(r'C:\Users\alex\workspace\housingprices\train.csv',index_col='Id')

train = train[train['GrLivArea']<4000] # remove outliers
train = train.fillna(train.mean())  #fill null values
numerics = train.dtypes[train.dtypes!='object']  #transform the numeric variables
normalized = np.log1p(train[numerics.index])
normalized = train[numerics.index]
categories = train.dtypes[train.dtypes=='object'] #dummify the categorical variables
categoricals = pd.get_dummies(train[categories.index])

combined = pd.concat([normalized,categoricals],axis=1) 
y = combined['SalePrice']
x = combined.drop('SalePrice',axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

linreg = LinearRegression() #fit the data to a model and find error

linreg.fit(X_train,y_train)
linreg.predict(X_test)
print("Score: {}".format(linreg.score(X_test,y_test)))
print("CV RMSE {}".format(np.sqrt(-cross_val_score(linreg,X_test,y_test,scoring="neg_mean_squared_error",cv=5))))


