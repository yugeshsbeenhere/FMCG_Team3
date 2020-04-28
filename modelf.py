
import pandas as pd
import numpy as np
import pickle


#LOAD THE DATASET
train_data = pd.read_csv("C:/Users/ADMIN/Desktop/excelR project/train1.csv")


#droping ['id'] column from train data
train_data.drop(["Unnamed: 0"], axis = 1, inplace = True)

#droping ['id'] column from train data
#test_data.drop(["Unnamed: 0"], axis = 1, inplace = True)
train_data['PROD_CD'] = train_data['PROD_CD'].str.replace(r'\D', '').astype(int)
train_data['SLSMAN_CD'] = train_data['SLSMAN_CD'].str.replace(r'\D', '').astype(int)
train_data['TARGET_IN_EA'] = train_data['TARGET_IN_EA'].str.replace(r'\D', '').astype(int)
train_data['ACH_IN_EA'] = train_data['ACH_IN_EA'].str.replace(r'\D', '').astype(int)


X = train_data.iloc[:, 0:4].values  
y = train_data.iloc[:, 4].values  


from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.ensemble import RandomForestRegressor

y_train=pd.DataFrame(y_train)

from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor

rf_multioutput = MultiOutputRegressor(ensemble.RandomForestRegressor(n_estimators=100,random_state = 15325))

rf_multioutput.fit(X_train, y_train)
# Saving model to disk
pickle.dump(rf_multioutput, open('model.pkl','wb'))



