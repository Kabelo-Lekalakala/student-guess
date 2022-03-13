import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


#organising of data 

data= pd.read_csv('student-mat.csv', sep=";")
data= data[["G1","G2","G3","failures","studytime","absences","famrel","activities"]]
dummies= pd.get_dummies(data.activities)
merged= pd.concat([data,dummies],axis='columns')
data= merged.drop(['activities'],axis='columns')
X= data.drop(columns=['G3'])
Y= data['G3']

#test training data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.10)

#predicting 

sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.fit_transform(X_test) 
regressor= linear_model.LinearRegression()
regressor.fit(X_train,Y_train)
scoreing= regressor.score(X_test,Y_test)

predictions= regressor.predict(X_test)



#visualization
plt.title("Final Grades")
plt.ylabel("Grades")
plt.xlabel("Student")
plt.xlim(right=40)
plt.plot(Y)
plt.plot(predictions)
plt.legend(["Final marks","Predictions"])
print(scoreing)
plt.show()

#p= 'absences'
#plt.scatter(data["G3"],data[p])