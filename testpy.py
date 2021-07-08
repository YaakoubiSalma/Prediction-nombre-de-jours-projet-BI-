import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

df = pd.read_csv('E:/Data Sience/Semestre 2/Projet BI/test_j.csv')
print(df)

# Data.to_excel(r'E:/Data Sience/Semestre 2/Projet BI/data.xlsx', index=False, header=True)

age = { '0-10':1, '21-30': 2,'31-40':3,'41-50':4,'51-60':5,'61-70':6,'71-80':7,'81-90':8,'81-90':9,'91-100':10   }
df['Age'] = df['Age'].map(age)
df['Age'].value_counts()
df.Age = df.Age.fillna(0)

Department = { 'gynecology':1, 'anesthesia':2, 'radiotherapy':3, ' TB & Chest disease':4}
df['Department'] = df['Department'].map(Department)
df['Department'].value_counts()
df.Department = df.Department.fillna(0)

Ward_Type = { 'R':1, 'Q':2, 'S':3, ' P':4, 'T':5, 'U':6}
df['Ward_Type'] = df['Ward_Type'].map(Ward_Type)
df['Ward_Type'].value_counts()
df.Ward_Type = df.Ward_Type.fillna(0)

Ward_Facility_Code = { 'F':1, 'E':2, 'D':3, ' B':4, 'C':5, 'A':6}
df['Ward_Facility_Code'] = df['Ward_Facility_Code'].map(Ward_Facility_Code)
df['Ward_Facility_Code'].value_counts()
df.Ward_Facility_Code = df.Ward_Facility_Code.fillna(0)

Hospital_region_code = { 'X':1, 'Y':2, 'Z':3}
df['Hospital_region_code'] = df['Hospital_region_code'].map(Hospital_region_code)
df['Hospital_region_code'].value_counts()
df.Hospital_region_code = df.Hospital_region_code.fillna(0)

Hospital_type_code = { 'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7}
df['Hospital_type_code'] = df['Hospital_type_code'].map(Hospital_type_code)
df['City_Code_Patient'].value_counts()
df.City_Code_Patient = df.City_Code_Patient.fillna(0)

stay = { '0-10':1, '21-30': 2,'31-40':3,'41-50':4,'51-60':5,'61-70':6,'71-80':7,'81-90':8,'81-90':9,'91-100':10   }
df['Stay'] = df['Stay'].map(age)
df['Stay'].value_counts()
df.Stay = df.Stay.fillna(0)

df['case_id'].value_counts()
df.case_id = df.case_id.fillna(0)

df['Hospital_code'].value_counts()
df.Hospital_code = df.Hospital_code.fillna(0)

df['City_Code_Hospital'].value_counts()
df.City_Code_Hospital = df.City_Code_Hospital.fillna(0)
 
df['patientid'].value_counts()
df.patientid = df.patientid .fillna(0)

df['Admission_Deposit'].value_counts()
df.Admission_Deposit = df.Admission_Deposit .fillna(0)

df=df.drop('Available Extra Rooms in Hospital', axis=1)
df=df.drop('Bed Grade', axis=1)
df=df.drop('Type of Admission', axis=1)
df=df.drop('Severity of Illness', axis=1)
df=df.drop('Visitors with Patient', axis=1)
df=df.drop('case_id', axis=1)
df=df.drop('patientid', axis=1)
df=df.drop('Admission_Deposit', axis=1)
df=df.drop('Hospital_type_code', axis=1)

target = df['Stay']
y = target 

# X1=df['Stay']
# le=LabelEncoder()
# X1new=le.fit_transform(X1)
# X1new=X1new.reshape(-1,1)

# X2=df['Age']
# le=LabelEncoder()
# X2new=le.fit_transform(X2)
# X2new=X2new.reshape(-1,1)

Data=df.drop('Stay', axis=1)

# Datanew=np.concatenate((Data,y),axis=1)
X=Data
y2=set(Data)

"4)Random forest

model=RandomForestRegressor(random_state=0, n_estimators=100)
model=LinearRegression()

import time
debut=time.time()
model.fit(Data, y)
fin=time.time()-debut
ypred=model.predict(Data)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y,ypred)
import math 
print('RMSE', math.sqrt(mse))

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(y,ypred)
print("Explained variance : %f" % (EV))

# "5)Decision tree
# from sklearn.model_selection import train_test_split
# x_train1,x_test1,train_Label,test_Label = train_test_split(Data, y, test_size=0.33, random_state=0)

# from sklearn.tree import DecisionTreeClassifier
# model=DecisionTreeClassifier(criterion='gini')
# model.fit(Data,y)
# pred=model.predict(x_test1)

# from sklearn.metrics import accuracy_score
# Acc=accuracy_score(test_Label, pred)*100
# print('accuracy score est=')
# print(Acc)

# from sklearn.metrics import confusion_matrix
# CM=confusion_matrix(test_Label, pred)
# print(CM)

"6)XGBoost
import xgboost as xgb
model=xgb.XGBRegressor(objective='reg:linear', linearing_rate=0.3, n_estimators=110)

import time
debut=time.time()
model.fit(Data, y)
fin=time.time()-debut
pred=model.predict(Data)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y,pred)
import math
print('RMSE', math.sqrt(mse))

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(y,pred)
print("Explained variance : %f" % (EV))

"7)Regression linear multiple
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(Data,y)

a=model.coef_
print(a)

b=model.intercept_
print(b)
X=np.array(Data)

pred=model.predict(Data)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y,pred)
import math 
print('RMSE', math.sqrt(mse))

import pandas as pd
Y1=pd.DataFrame(y)
Y1.describe()

from sklearn.metrics import explained_variance_score
EV=explained_variance_score(y,pred)
print("Explained variance : %f" % (EV))

from sklearn.metrics import r2_score
r=r2_score(y,pred)
print(r)