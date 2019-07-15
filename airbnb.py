#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:17:59 2019

@author: niloofarzarifi
"""
#airbnb-Toronto
""

#http://tomslee.net/airbnb-data-collection-get-the-data
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import os

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from random import randint
import sklearn
print (sklearn.__version__) #0.19.1
os.getcwd()
os.chdir('/Users/niloofarzarifi/Desktop/Udacity/khaneh/airbnb-Toronto/')

df1 = pd.read_csv('./2017/tomslee_airbnb_toronto_0778_2017-01-14.csv')
df2 = pd.read_csv('./2017/tomslee_airbnb_toronto_0857_2017-02-16.csv')
df3 = pd.read_csv('./2017/tomslee_airbnb_toronto_0930_2017-03-12.csv')
df4 = pd.read_csv('./2017/tomslee_airbnb_toronto_1042_2017-04-08.csv')
#------------------------------------
frames = [df1, df2,df3,df4]
result=pd.concat(frames)
result.shape

DF=result[result.duplicated()]
result[result.duplicated(['host_id'])].count()
result[result.duplicated(['room_id'])].count()  
result[result.duplicated(['last_modified'])].count() 
result.duplicated(['room_id', 'host_id','room_type', 'neighborhood']) #True or False

result.isnull().sum()  #drop two empty cols
result.columns.tolist()
result=result.drop(['borough','minstay','room_id','host_id','latitude','longitude','last_modified'],axis=1)

result.columns.tolist()
 
result.head()
result.overall_satisfaction.value_counts()
result.overall_satisfaction.nunique() #target is 10 classes
result.room_type.value_counts()  #make dummy
result.neighborhood.value_counts() #139 neighborhood

df=pd.get_dummies(result,columns=['room_type','neighborhood'],drop_first=True)
df.shape
df.to_csv("cleaned-airbnb.csv",index=False)
#--------------

data=pd.read_csv('./cleaned-airbnb.csv')
data.head() #145 columns
data.shape

#must convert float to int (cannot convert to object, err:unkown format in KNN model )
#otherwise it gets continus error in classification 
#old-class:0,1,1.5,2,2.5,3,3.5,4,4.5,5
#new-class:0,2,3,4,5,6,7,8,9,10
y=data['overall_satisfaction']*2  
#int format converts 4.5, 3.5 that's why I changed it to 2,3,4,...10 
y=y.astype(int) 
y.dtypes
type(y)
y.value_counts()

X=data.drop(['overall_satisfaction'],axis=1)
#only interested to use z-score for price col, 
#StandardScaler will change all dummies to continus number which decreases the accuracy score by 3%
X['price']= (data['price'] - data['price'].mean())/data['price'].std(ddof=0)


"""
#prefer to not use standardscalar
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_scaled=sc.fit_transform(X)
type(X_scaled)
"""
#---------------------------------------
filter_col=X.columns.tolist()
len(filter_col)

#RandomForest cross validation with 10 Kfold  

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image
from sklearn import model_selection
#model.get_params()




Mean_result=[]
Full_result=[]
kfold = model_selection.KFold(n_splits=10, random_state=0)
#optimize number of trees
for num_trees in range(3,20):
    model = RandomForestClassifier(n_estimators=num_trees)
    results = model_selection.cross_val_score(model, X, y, cv=kfold)
    Mean_result.append(results.mean())
    Full_result.append(results)
    print("RandomForestClassifier, Cross_val_score=",results.mean() )
    #
#---------------------------------

plt.plot(range(3,20),Mean_result)
plt.scatter(range(3,20),Mean_result)
plt.xlabel("number of trees")
plt.ylabel('Cross_val_score')
plt.title('Mean of Cross validation scores with 10 Kfolds')

#here I show why we need results.mean()
lt=[range(3,20)] * 10
plt.scatter(lt,Full_result)
plt.xlabel("number of trees")
plt.ylabel('Cross_val_score')
plt.title('Cross validation scores with 10 Kfolds')


#how about Max_depth?
#How to Visualize a Decision Tree from a Random Forest in Python using Scikit-Learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y ,random_state=0)    
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image
model = RandomForestClassifier(n_estimators=10,max_depth=20) #max_depth=None takes forever

# Train
model.fit(X_train, y_train)
# Extract single tree
estimator = model.estimators_[3] #we can change the tree number:) here I show #3

# Create DOT data
dot_data = tree.export_graphviz(estimator, out_file=None, 
                                feature_names=filter_col,  
                                class_names=['0','2','3','4','5','6','7','8','9','10'],
                                rounded = True, proportion = False,
                                precision = 2, filled = True)
                                

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  
# Show graph
Image(graph.create_png())

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

confusion_matrix(y_test,y_pred)

print classification_report(y_test,y_pred)

accuracy_score(y_test,y_pred) #The accuracy drops with max_depth=20, therefore 90% score occurs with much higher depth
#let's try to optimize the RF with gridsearch

#This is the default values
model = RandomForestClassifier(random_state = 0)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(model.get_params())

#Random Hyperparameter Grid
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators=[10,11,12,25,15]
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth=[10,20,30]
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [10, 20, 30]
# Minimum number of samples required at each leaf node
min_samples_leaf = [5, 10, 15]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)



# Use the random grid to search for best hyperparameters
# First create the base model to tune
model = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
#More n_iter will cover a wider search space and more cv folds reduces the chances of overfitting
# Fit the random search model
model_random.fit(X_train, y_train)
model_random.best_params_


model = RandomForestClassifier(n_estimators=12,max_depth=30,max_features='sqrt',min_samples_leaf=5,min_samples_split=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)  #0.78 with default values the score was 0.79!


model = RandomForestClassifier(n_estimators=12,max_depth=30)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred) #0.83 the accuracy improves if we use default setting
#performance vs time is one of the most fundamental which one comes first!? 

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
i = pd.DataFrame({'feature': filter_col,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)

#let's ignore neighborhood
XX=X[['reviews','accommodates','bedrooms','price','room_type_Private room','room_type_Shared room']]
XX.head()
XX_train, XX_test, y_train, y_test = train_test_split(XX, y, test_size=0.3, stratify=y ,random_state=0)
#
#
#simpler model with 4% lower accuracy
model = RandomForestClassifier(n_estimators=12,max_depth=30)
model.fit(XX_train, y_train)
y_pred = model.predict(XX_test)
accuracy_score(y_test,y_pred)

# Probabilities for each class
model_probs = model.predict_proba(XX_test)[:, 1]

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)

score = [] #to store rmse values for different k
for K in range(5):
    K = K+1
    model = KNeighborsClassifier(n_neighbors = K)
    
    model.fit(X_train, y_train)  #fit the model
    y_pred=model.predict(X_test) #make prediction on test set
    score.append(accuracy_score(y_test,y_pred))
    cur_score=accuracy_score(y_test,y_pred)
    print('score value for k= ' , K , 'is:', cur_score)
    
    
#KNN performs better with a lower number of features 
#let's ignore neighborhood
score = [] #to store accuracy values for different k
for K in range(10):
    K = K+1
    model = KNeighborsClassifier(n_neighbors = K)

    model.fit(XX_train, y_train)  #fit the model
    y_pred=model.predict(XX_test) #make prediction on test set
    score.append(accuracy_score(y_test,y_pred))
    cur_score=accuracy_score(y_test,y_pred)
    print('score value for k= ' , K , 'is:', cur_score)
 #might be overfitting problem   

# test the training set
score_trained = [] 
for K in range(10):
    K = K+1
    model = KNeighborsClassifier(n_neighbors = K)

    model.fit(XX_train, y_train)  #fit the model
    y_pred=model.predict(XX_train) #make prediction on test set
    score_trained.append(accuracy_score(y_train,y_pred))
    cur_score_trained=accuracy_score(y_train,y_pred)
    print('score value for k= ' , K , 'is:', cur_score_trained)
 #over 10% difference, what it tells? 
 
ax = plt.subplot(111)
plt.plot(range(1,11),score,label='validation')
plt.plot(range(1,11),score_trained,color='red',label='trained score')
plt.xlabel("number of K")
plt.legend()
plt.ylabel('Cross_val_score')
plt.title('The accuracy scores,KNN')
#over 10% difference, what it tells?  a poor model? overfitting?

 from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection
# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in range(20):
    k=k+1
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = model_selection.cross_val_score(knn, XX, y, cv=10,scoring='accuracy' )
    
    cv_scores.append(scores.mean())
    print('cross_val_score: ' , k , 'is:', scores.mean())
    
plt.plot(range(1,21),cv_scores)

plt.xlabel("number of K")
plt.ylabel('Cross_val_score')
plt.title('Mean of Cross validation scores with cv=10')
#with 70% of data trained, we have overfitting issue 


# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = range(20)[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k

# plot misclassification error vs k
plt.plot(range(20), MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()



    
 
    





























































    






























