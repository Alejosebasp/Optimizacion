
import numpy as np
import random

train_data = np.load("dataset\DB_Vecs_100.npy")
train_label = np.load("dataset\DB_Labels_100.npy")
test_data = np.load("dataset\Q_vecs_100.npy")
train_label = train_label.reshape((train_label.shape[0],1))

data_label = np.concatenate((train_data,train_label),axis = 1)
uni= np.unique(data_label, axis=0)
train_data = uni[uni[:,-1].argsort()]

train_data = np.delete(train_data, -1, 1)
train_label = np.array(uni[:,-1]).reshape((train_data.shape[0],1))

train_label[train_label==0] = -1

no_train_sample = train_data.shape[0]
no_variable = train_data.shape[1] + 1

c = [0]*(no_variable*2) + [-1]*no_train_sample + [0]*no_train_sample
b = [-1]*no_train_sample


U = np.diag([-1]*no_train_sample)
S = np.diag([1]*no_train_sample)
A = np.zeros((no_train_sample,no_variable*2))
one = np.array([random.gauss(1, 0.1) for i in range(no_train_sample)]).reshape((no_train_sample,1))
Z = train_label*np.concatenate((train_data,one),axis = 1)

for i in range(no_variable):
    A[:,2*i] = Z[:,i] 
    A[:,2*i+1] = -1*Z[:,i] 

A = np.concatenate((A,U),axis = 1)
A = np.concatenate((A,S),axis = 1)