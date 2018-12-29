# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 22:47:54 2018

@author: YASH SAINI
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

col=['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion'
     ,'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli',
     'Mitoses','Class']

df=pd.read_csv('breast-cancer-wisconsin.data.csv',names=col,header=None)

#Data Preprocessing
df['Bare Nuclei'].replace('?',np.NAN,inplace=True)
df=df.dropna()

df['Class']=df['Class']/2 -1

X=df.drop(['id','Class'],axis=1)
X_col=X.columns

Y=df['Class']

#Scaling
X=StandardScaler().fit_transform(X.values)

df1=pd.DataFrame(X,columns=X_col)
# or scale using minmax ,both result in different data
'''from sklearn.preprocessing import MinMaxScaler
pd.DataFrame(MinMaxScaler().fit_transform(df.drop(['id','Class'],axis=1).values),columns=X_col)
'''
#Splitting of Data
X_train,X_test,Y_train,Y_test= train_test_split(df1,Y,train_size=0.8,random_state=42)


"""KNN CLassifier"""
#Model fitting & Predicting
knn= KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(X_train,Y_train)

def print_score(clf,X_train,Y_train,X_test, Y_test,train=True):
    if train:
        print("\nTraining Result:")
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_train,clf.predict(X_train))))
        print("\nClassification_report: \n{}\n".format(classification_report(Y_train,clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(Y_train,clf.predict(X_train))))
        
        res=cross_val_score(clf,X_train,Y_train,cv=10,scoring="accuracy")
        print("Average Accuracy:\t {0:.4f}".format(np.mean(res)))
        print ("Accuracy SD:\t\t {0:.4f}".format(np.std(res)))
    
    elif train==False:
        print("\nTest Results:")
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_test,clf.predict(X_test))))
        print("\nClassification_report: \n{}\n".format(classification_report(Y_test,clf.predict(X_test))))
        print("\nConfusion Matrix:\n{}\n".format(confusion_matrix(Y_test,clf.predict(X_test))))
 
print ("\n KNN Classifier")       
#Scores for training data
print_score(knn,X_train,Y_train,X_test, Y_test,train=True)
#Scores for test data
print_score(knn,X_train,Y_train,X_test, Y_test,train=False)

'''#Improving the results we use grid search 
#It takes time , we find almost matching accurcies but accuracy of neighbours=7 is the best
params={'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
grid_search_cv=GridSearchCV(KNeighborsClassifier(),params,n_jobs=-1,verbose=1)
grid_search_cv.fit(X_train,Y_train)
best=grid_search_cv.best_estimator_
print (grid_search_cv.cv_results_['mean_train_score'])
print (grid_search_cv.cv_results_)

print_score(grid_search_cv,X_train,Y_train,X_test, Y_test,train=True)

'''


""" SVM"""
clf=svm.SVC(kernel='rbf', degree=3,  gamma=0.7)
clf.fit(X_train,Y_train)

print("\n\n SVM Details\n\n")
#Scores for training data
print_score(clf,X_train,Y_train,X_test, Y_test,train=True)
#Scores for test data
print_score(clf,X_train,Y_train,X_test, Y_test,train=False)



""" Random Forest"""
clf=RandomForestClassifier(random_state=42)
clf.fit(X_train,Y_train)

print("\n\n Random Forest Classifier\n\n")
#Scores for training data
print_score(clf,X_train,Y_train,X_test, Y_test,train=True)
#Scores for test data
print_score(clf,X_train,Y_train,X_test, Y_test,train=False)


"""XG BOOST"""
import xgboost as xgb
clf=xgb.XGBClassifier()
clf.fit(X_train,Y_train)
print("\n\n XG BOOST\n\n")
#Scores for training data
print_score(clf,X_train,Y_train,X_test, Y_test,train=True)
#Scores for test data
print_score(clf,X_train,Y_train,X_test, Y_test,train=False)











