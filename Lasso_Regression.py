# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

# Loading Data
data = pd.read_csv('C:/Users/tiwar/OneDrive/Desktop/Spyder_projects/Machine_Learning/HousingData.csv')

print(data.head())

print(data.shape)

print(data.columns)

X = data.drop(['MEDV'],axis=1)
y = data['MEDV']

print(X.info())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)

# Feature engineering
knn = KNNImputer()

knn.fit(X_train)

X_train_trf = knn.transform(X_train)
X_test_trf = knn.transform(X_test)

pd.DataFrame(X_train_trf, columns=X_train.columns)
pd.DataFrame(X_test_trf, columns=X_train.columns)

scaler = StandardScaler()

scaler.fit_transform(X_train_trf)
scaler.transform(X_test_trf)

pd.DataFrame(X_train_trf, columns=X_train.columns)
pd.DataFrame(X_test_trf, columns=X_train.columns)
# We can reverse the Standard scaler using => scaler.inverse_transform

# Training model
clf = LinearRegression()

mean_square_error = cross_val_score(clf, X_train_trf, y_train, scoring='neg_mean_squared_error', cv=10)

print("Final mean square error:",np.mean(mean_square_error))

clf.fit(X_train_trf, y_train)

y_pred = clf.predict(X_test_trf)

sns.distplot(y_pred-y_test, kde={'alpha':0.5})
# Variance is -10 to 10 that means our model performed very well

print("R^2Score:",r2_score(y_test, y_pred))

# Lasso Regression
lasso_regression = Lasso()

perameters = {'alpha':[1,2,3,4,5,6,7,8,9,10,11]}
lassocv = GridSearchCV(lasso_regression, perameters, scoring='neg_mean_squared_error',cv=5)

lassocv.fit(X_train_trf, y_train)

print(lassocv.best_params_)
print(lassocv.best_score_)

y_new_pred = lassocv.predict(X_test_trf)

sns.distplot(y_new_pred-y_test, kde={'alpha':0.5})

print("R^2Score:",r2_score(y_test, y_new_pred))
