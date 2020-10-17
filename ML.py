###  download data from
###  http://archive.ics.uci.edu/ml/datasets/Adult.

##Here we are using dataset that contains the information about individuals from various countries.
# Our target is to predict whether a person makes <=50k or >50k annually on basis of the other information available.
# Dataset consists of 32561 observations and 14 features describing individuals.


#importing standard libraries
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import datetime as dt

#import lightgbm and xgboost
import lightgbm as lgb
#import xgboost as xgb

#loading our training dataset 'adult.csv' with name 'data' using pandas
data=pd.read_csv(r'C:\Users\JCAO\Hisotrical_Works\Module_3_Python\adult.txt',header=None)

#Assigning names to the columns
data.columns=['age','workclass','fnlwgt','education','education-num','marital_Status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','Income']

#glimpse of the dataset
print(data.columns)

# Label Encoding our target variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l=LabelEncoder()
l.fit(data.Income)

print(l.classes_)
data.Income=Series(l.transform(data.Income))  #label encoding our target variable
data.Income.value_counts()



#One Hot Encoding of the Categorical features
one_hot_workclass=pd.get_dummies(data.workclass)
one_hot_education=pd.get_dummies(data.education)
one_hot_marital_Status=pd.get_dummies(data.marital_Status)
one_hot_occupation=pd.get_dummies(data.occupation)
one_hot_relationship=pd.get_dummies(data.relationship)
one_hot_race=pd.get_dummies(data.race)
one_hot_sex=pd.get_dummies(data.sex)
one_hot_native_country=pd.get_dummies(data.native_country)

#removing categorical features
data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True)



#Merging one hot encoded features with our dataset 'data'
data=pd.concat([data,one_hot_workclass,one_hot_education,one_hot_marital_Status,one_hot_occupation,one_hot_relationship,one_hot_race,one_hot_sex,one_hot_native_country],axis=1)

print(data.head())

#removing dulpicate columns
_, i = np.unique(data.columns, return_index=True)
data=data.iloc[:, i]

#Here our target variable is 'Income' with values as 1 or 0.
#Separating our data into features dataset x and our target dataset y
x=data.drop('Income',axis=1)
y=data.Income
labels = list(set(y))
headers = x.columns.tolist()

#Imputing missing values in our target variable
y.fillna(y.mode()[0],inplace=True)

#Now splitting our dataset into test and train
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)



### start apply light GBM
train_data=lgb.Dataset(x_train,label=y_train)
#setting parameters for lightgbm
param = {'num_leaves':150, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']
#Here we have set max_depth in xgb and LightGBM to 7 to have a fair comparison between the two.
#training our model using light gbm
num_round=50
start=dt.datetime.now()
lgbm=lgb.train(param,train_data,num_round)
stop=dt.datetime.now()
#Execution time of the model
execution_time_lgbm = stop-start
print(execution_time_lgbm)
#predicting on test set
ypred2=lgbm.predict(x_test)
print(ypred2[0:5])  # showing first 5 predictions
#converting probabilities into 0 or 1
for i in range(0,9769):
    if ypred2[i]>=0.5:       # setting threshold to .5
       ypred2[i]=1
    else:
       ypred2[i]=0
#calculating accuracy
accuracy_lgbm = accuracy_score(ypred2,y_test)
print(accuracy_lgbm)
y_test.value_counts()
from sklearn.metrics import roc_auc_score
#calculating roc_auc_score for xgboost

#calculating roc_auc_score for light gbm.
auc_lgbm = roc_auc_score(y_test,ypred2)
print(auc_lgbm)
#comparison_dict = {'accuracy score':(accuracy_lgbm),'auc score':(auc_lgbm)}

#comparison_df = DataFrame(comparison_dict)
#comparison_df.index= ['LightGBM','xgboost']
#print(comparison_df)
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
dTree = DecisionTreeClassifier(max_depth=7,criterion='entropy',random_state=0)
dTree.fit(x_train,y_train)
y_pred_tree = dTree.predict(x_test)
accuracy_tree = accuracy_score(y_pred_tree,y_test)

print('light BGM Accuracy rate is '+str(accuracy_lgbm)+'decision tree Accuracy rate is'  + str(accuracy_tree))

###please note the data table is one hot encoded
# import graphviz
#
# print(headers)
# print(len(headers))
# print(labels)
#
# dot_data = tree.export_graphviz(dTree,
#                                      out_file=None,
#                                      feature_names=headers,
#                                      class_names=labels,
#                                      filled=True,
#                                      rounded=True,
#                                      special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render('dtree_render',view=True)
# graph.view()

def plot_tree():
 from sklearn import tree
 import graphviz


 dot_data = tree.export_graphviz(clf,
                                     out_file=None,
                                     feature_names=headers,
                                     class_names=labels,
                                     filled=True,
                                     rounded=True,
                                     special_characters=True)
 graph = graphviz.Source(dot_data)
 graph.render('dtree_render',view=True)
 graph.view()
