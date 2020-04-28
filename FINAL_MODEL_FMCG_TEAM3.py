import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#LOAD THE DATASET
train_data = pd.read_csv("C:/Users/ADMIN/Desktop/excelR project/train1.csv")
test_data = pd.read_csv("C:/Users/ADMIN/Desktop/excelR project/test1.csv")

#getting the discriptive information about the train data
train_data.head(10) 
train_data.shape

#droping ['id'] column from train data
train_data.drop(["Unnamed: 0"], axis = 1, inplace = True)
print(train_data.keys())

test_data.drop(["Unnamed: 0"], axis = 1, inplace = True)
print(test_data.keys())

#also droping year column 
train_data.drop(["PLAN_YEAR"], axis = 1, inplace = True)
print(train_data.keys())

test_data.drop(["PLAN_YEAR"], axis = 1, inplace = True)
print(test_data.keys())

#ENCODING  (using this we have remove string part and kept integer in following colunms for EDA and better insights )
train_data['PROD_CD'] = train_data['PROD_CD'].str.replace(r'\D', '').astype(int)
train_data['SLSMAN_CD'] = train_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
train_data['TARGET_IN_EA'] = train_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)
train_data['ACH_IN_EA'] = train_data['ACH_IN_EA'].str.replace(r'\D', '').astype(int)

test_data['PROD_CD'] = test_data['PROD_CD'].str.replace(r'\D', '').astype(int)
test_data['SLSMAN_CD'] = test_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
test_data['TARGET_IN_EA'] = test_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)

# here we are droping rows in which value is zero for traget and achievment

indexNames = train_data[ (train_data['TARGET_IN_EA'] == 0) & (train_data['ACH_IN_EA'] == 0) ].index
train_data.drop(indexNames , inplace=True)
#droping 1st month
train_data=train_data[train_data["PLAN_MONTH"]>1]  #droping 1st month
#test_data=test_data[test_data["PLAN_MONTH"]>1]

X = train_data.iloc[:, 0:4].values  
y = train_data.iloc[:, 4].values  

from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

#rf_multioutput = MultiOutputRegressor(ensemble.RandomForestRegressor(n_estimators=500, n_jobs=1, verbose=1))
rf_multioutput = MultiOutputRegressor(ensemble.RandomForestRegressor(n_estimators=100, random_state=15325))
#lgb_multioutput = MultiOutputRegressor(lgb.LGBMRegressor(learning_rate=0.05,max_depth=7,n_jobs=1,n_estimators=1000,nthread=-1))
rf_multioutput.fit(X_train, y_train)

rf_train_RMSE=np.mean((rf_multioutput.predict(X_train) - y_train)**2, axis=0)
rf_train_RMSE = np.sqrt(rf_train_RMSE)
print(rf_train_RMSE)

from sklearn.metrics import r2_score
rf_train_pred = rf_multioutput.predict(X_train)
rf_train_R2 = r2_score(y_train, rf_train_pred)

rf_multioutput.fit(X_test, y_test)
rf_test_RMSE=np.mean((rf_multioutput.predict(X_test) - y_test)**2, axis=0)
rf_test_RMSE= np.sqrt(rf_test_RMSE)
print(rf_test_RMSE)

rf_test_pred = rf_multioutput.predict(X_test)
rf_test_R2 = r2_score(y_test, rf_test_pred)#0.9990043465338014

#fitting test data on our model 

y_prednew = rf_multioutput.predict(test_data)
test_data = pd.read_csv("C:/Users/ADMIN/Desktop/excelR project/test1.csv")

test_data['ACH_IN_EA']=y_prednew
#rounding off the values
decimals = 0    
test_data['ACH_IN_EA'] = test_data['ACH_IN_EA'].apply(lambda x: round(x, decimals))

#test_data.to_csv("test_data.csv",encoding="utf-8")
test_data.to_csv("C:/Users/ADMIN/Desktop/excelR project/testdata.csv")
