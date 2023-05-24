# Linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('C:/Users/tiwar/OneDrive/Desktop/Spyder_projects/100-days-of-machine-learning-main/day48-simple-linear-regression/placement.csv')
print(data.head())

plt.scatter(data['cgpa'],data['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(In LPA')

X = data[['cgpa']]
y = data['package']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = LinearRegression()

clf.fit(X_train,y_train)

plt.scatter(data['cgpa'],data['package'])
plt.plot(X_test, clf.predict(X_test), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(In LPA')

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))