#---------------------------------------------------------------------------------------
#---Prediction of Relative Humidity ----------------------------------------------------
#---------------------------------------------------------------------------------------

#---Working Directory-------------------------------------------------------------------
import os
os.chdir(...)
os.getcwd()

#---Data Loading------------------------------------------------------------------------
import pandas as pd

df = pd.read_csv("Air Quality UCI.csv",na_values=-200) # -200 was found while looking at the dataset

df.head(5)
df.tail(5)
df.columns
df.isnull().sum()
df.info()
df.shape

#---Data Cleaning-----------------------------------------------------------------------
df.dropna(thresh=7,axis=0,inplace=True)
df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)
df.drop('NMHC(GT)', axis=1, inplace=True) # Most of the values are missing

df['Date']=pd.to_datetime(df.Date, format='%m/%d/%Y')  # Format date column
df.set_index('Date',inplace=True)

df['Month']=df.index.month     #Create nonth column
df.reset_index(inplace=True)

df['Hour']=df['Time'].apply(lambda x: int(x.split(':')[0])) # Split hour from time

#---Replacinig NaN values with particular hour average in each month--------------------
df['CO(GT)']=df['CO(GT)'].fillna(df.groupby(['Month','Hour'])['CO(GT)'].transform('mean'))
df['NOx(GT)']=df['NOx(GT)'].fillna(df.groupby(['Month','Hour'])['NOx(GT)'].transform('mean'))
df['NO2(GT)']=df['NO2(GT)'].fillna(df.groupby(['Month','Hour'])['NO2(GT)'].transform('mean'))

df.isnull().sum()

#---Replacing what is left by average of coresponding hour------------------------------
df['CO(GT)']=df['CO(GT)'].fillna(df.groupby(['Hour'])['CO(GT)'].transform('mean'))
df['NOx(GT)']=df['NOx(GT)'].fillna(df.groupby(['Hour'])['NOx(GT)'].transform('mean'))
df['NO2(GT)']=df['NO2(GT)'].fillna(df.groupby(['Hour'])['NO2(GT)'].transform('mean'))

df.isnull().sum()

#---Correlation-------------------------------------------------------------------------
from matplotlib import pyplot as plt
import seaborn as sns

correlation = df.corr()
correlation['RH'].sort_values(ascending=False)

plt.figure(figsize=(14,8))
sns.heatmap(df.corr(),annot=True,cmap='viridis')

#---Splitting and Scaling---------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[:].drop(['RH', 'Date', 'Time'] ,axis=1)
y = df['RH']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#---------------------------------------------------------------------------------------
#---Linear Regressin Model--------------------------------------------------------------
#---------------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

#---Visualization-----------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

y_pred = lr.predict(X_test)
rmse = round(np.sqrt(mean_squared_error(y_test,y_pred)),2)

plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Linear Regression')
plt.show()


#---GridSearch--------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'fit_intercept':[True, False], 'normalize':[True, False]}

GS = GridSearchCV(lr, parameters,cv=5)
GS.fit(X_train,y_train)

GS.best_params_
GS.best_score_

#---------------------------------------------------------------------------------------
#---Random Forest-----------------------------------------------------------------------
#---------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 0)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

#---Visualization-----------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

y_pred = rf.predict(X_test)
rmse = round(np.sqrt(mean_squared_error(y_test,y_pred)),2)

plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Random Forests')
plt.show()

#---GridSearch--------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [10,20,30],
              'max_features':['auto', 'sqrt', 'log2'],
             }

GS = GridSearchCV(rf, parameters,cv=5)
GS.fit(X_train,y_train)

GS.best_params_
GS.best_score_

#---------------------------------------------------------------------------------------
#---SVR---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train,y_train)
svr.score(X_test,y_test)

#---Visualization-----------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

y_pred = svr.predict(X_test)
rmse = round(np.sqrt(mean_squared_error(y_test,y_pred)),2)

plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('SVR')
plt.show()

#---GridSearch--------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['rbf', 'sigmoid'],
              'degree':[2,3,4]
             }

GS = GridSearchCV(svr, parameters,cv=5)
GS.fit(X_train,y_train)

GS.best_params_
GS.best_score_

#---------------------------------------------------------------------------------------
#---Polynomial Regression---------------------------------------------------------------
#---------------------------------------------------------------------------------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

polyfeatures = PolynomialFeatures()
X_poly = polyfeatures.fit_transform(X)
X_poly = X_poly[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X_poly,y,random_state=0)

polyreg = LinearRegression()
polyreg.fit(X_train, y_train)
polyreg.score(X_test,y_test)

#---Visualization-----------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

y_pred = polyreg.predict(X_test)
rmse = round(np.sqrt(mean_squared_error(y_test,y_pred)),2)

plt.scatter(y_test,y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('Polynomial Regression')
plt.show()

#---GridSearch--------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

parameters = {'fit_intercept':[True, False], 'normalize':[True, False]}

GS = GridSearchCV(polyreg, parameters,cv=5)
GS.fit(X_train,y_train)

GS.best_params_
GS.best_score_

#---End---------------------------------------------------------------------------------

