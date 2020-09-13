import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('apy.csv')
print(data)
#pre-processing steps
data = data.dropna()

data = data.drop(['Crop_Year'],axis=1)
columns = data.columns

le = LabelEncoder()

for i in range(len(columns)):
    data[columns[i]] = le.fit_transform(data[columns[i]])

X = data.drop(['Production'],axis=1)
Y = data['Production']


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=0)


'''''#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
y_train = sc.fit_transform(y_train)
'''
# Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print("Predicted Value:",y_pred)


#K-Nearest Neighbors

print("********* K-NN Algorithm******")
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print("Crop Production:",pred)
accuracy = 100*round(accuracy_score(y_test,pred),3)
print("Accuracy Value of crop production is:",accuracy)

#Graphical Representation of Using Linear Regression
#print(y_pred.shape)
#print(y_test.shape)

plt.bar(y_pred, y_test)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title("Crop Prediction(Linear Regression)")
plt.show()

# Graphical Representation of Using KNN

plt.bar(pred,y_test)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title("Crop Prediction(KNN)")
plt.show()


