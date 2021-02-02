import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,ReLU,Activation,Flatten,BatchNormalization
import keras
from keras.activations import relu,sigmoid
from keras.wrappers.scikit_learn import KerasClassifier
import sklearn
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('C:/Users/Chirag Chauhan/Desktop/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]     

geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

X=pd.concat([X,geography,gender],axis=1)

X=X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##Using Grid SearchCV
def create_model(layers,activation):
    model = Sequential()
    for i ,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    model.add(Dense(units = 1,kernel_initializer='glorot_uniform',activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,verbose=0)

layers=[[20,],[40,20],[45,30,15]]
activations = ['sigmoid','relu']
param_grid = dict(layers=layers,activation=activations,batch_size=(128,256),epochs=[5])
grid = GridSearchCV(estimator = model,param_grid=param_grid,cv=5)
res = grid.fit(X_train,y_train)

best_params = grid.best_params_
best_scores = grid.best_score_



























