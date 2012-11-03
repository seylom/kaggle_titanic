'''
Created on Nov 1, 2012

@author: SeylomA
'''
 
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os 

def load_data():
    if os.path.exists("train.pkl"):
        train = joblib.load("train.pkl")
        np.savetxt("train_preprocessed.csv", train, delimiter=",")
    else:
        train = pd.read_csv("data/train.csv")
        train = np.array(train)
        train = process_data(train)
        # train = np.delete(train, [2, 7, 9], 1)  # remove these columns
        joblib.dump(train, "train.pkl")
        
    if os.path.exists("test.pkl"):
        test = joblib.load("test.pkl")
        np.savetxt("test_preprocessed.csv", test, delimiter=",")
    else:
        test = pd.read_csv("data/test.csv")
        test = np.array(test)
        test = process_data(test, False)
        # test = np.delete(test, [1, 6, 8], 1)  # remove these columns
        joblib.dump(test, "test.pkl")
        
    train_Y = np.asarray([x[0] for x in train])
    train_X = np.asarray([x[1:] for x in train])
    
    return train_X, train_Y, test

def process_data(data, is_train=True):
    
    ix = is_train and 1 or 0
    
    # Male = 1, female = 0:
    data[data[0::, 2 + ix] == 'male', 2 + ix] = 1
    data[data[0::, 2 + ix] == 'female', 2 + ix] = 0
    
    # embark c=0, s=1, q=2
    data[data[0::, 9 + ix] == 'C', 9 + ix] = 0
    data[data[0::, 9 + ix] == 'S', 9 + ix] = 1
    data[data[0::, 9 + ix] == 'Q', 9 + ix] = 2
    
    # All the ages with no data make the median of the data
    data[data[0::, 3 + ix] == '', 3 + ix] = np.median(data[data[0::, 3 + ix]\
                                               != '', 3 + ix].astype(np.float))
    # All missing ebmbarks just make them embark from most common place
    data[data[0::, 9 + ix] == '', 9 + ix] = np.round(np.mean(data[data[0::, 9 + ix]\
                                                       != '', 9 + ix].astype(np.float)))
    
    # add an adult/child column
    child = np.zeros((len(data), 1))
    child[data[0::, 3 + ix ] < 18] = 1;

    # add dummy variables for class
    class_c = np.zeros((len(data), 1))
    class_s = np.zeros((len(data), 1))
    class_q = np.zeros((len(data), 1))
    
    class_c[data[0::, 9 + ix] == 0] = 1
    class_s[data[0::, 9 + ix] == 1] = 1
    class_q[data[0::, 9 + ix] == 2] = 1
    
    train_data = np.delete(data, [1 + ix, 6 + +ix, 8 + ix], 1)  # remove the name data, cabin and ticket
    train_data = np.append(train_data, child, axis=1)
    
    train_data = np.append(train_data, class_c, axis=1)
    train_data = np.append(train_data, class_s, axis=1)
    train_data = np.append(train_data, class_q, axis=1)
    
    return train_data
    
