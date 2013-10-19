'''
Created on Nov 1, 2012

@author: SeylomA
'''
 
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os 
from sklearn.preprocessing import OneHotEncoder

def load_data_one_hot():
    
    save = False
    if os.path.exists("train_correct.pkl"):
        train = joblib.load("train_correct.pkl")
        #np.savetxt("train_preprocessed.csv", train, delimiter=",", fmt='%.4f')
    else:
        save = True
        train = pd.read_csv("data/train_clean_no_missing.csv")
        train = np.array(train)
        #train = process_data(train)
        # train = np.delete(train, [2, 7, 9], 1)  # remove these columns
       
        
    if os.path.exists("test_correct.pkl"):
        test = joblib.load("test_correct.pkl")
        #np.savetxt("test_preprocessed.csv", test, delimiter=",", fmt='%.4f')
    else:
        save = True
        test = pd.read_csv("data/test_clean_no_missing.csv")
        test = np.array(test)
        #test = process_data(test, False)
        # test = np.delete(test, [1, 6, 8], 1)  # remove these columns
      
    train_Y = np.asarray([x[0] for x in train])
    train_X = np.asarray([x[1:] for x in train]) 
    
#    encoder = OneHotEncoder()
#    encoder.fit(np.vstack((train_X,test)))
#    
#    train_X = encoder.transform(train_X)  # Returns a sparse matrix (see numpy.sparse)
#    test = encoder.transform(test)
    
    if save:
        joblib.dump(train, "train_onehot.pkl")  
        joblib.dump(test, "test_onehot.pkl")
    
    return train_X, train_Y, test

def load_data():
    if os.path.exists("train.pkl"):
        train = joblib.load("train.pkl")
        np.savetxt("train_preprocessed.csv", train, delimiter=",", fmt='%.4f')
    else:
        train = pd.read_csv("data/train.csv")
        train = np.array(train)
        train = process_data(train)
        # train = np.delete(train, [2, 7, 9], 1)  # remove these columns
        joblib.dump(train, "train.pkl")
        
    if os.path.exists("test.pkl"):
        test = joblib.load("test.pkl")
        np.savetxt("test_preprocessed.csv", test, delimiter=",", fmt='%.4f')
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
    data[data[0::, 3 + ix] == '', 3 + ix] = np.median(data[data[0::, 3 + ix] != '', 3 + ix].astype(np.float))
    # All missing ebmbarks just make them embark from most common place
    data[data[0::, 9 + ix] == '', 9 + ix] = np.round(np.mean(data[data[0::, 9 + ix] != '', 9 + ix].astype(np.float)))
    
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

def process_data2(data, is_train=True):
    
    ix = is_train and 1 or 0
    
    # Male = 1, female = 0:
    data[data[0::, 2 + ix] == 'male', 2 + ix] = 1
    data[data[0::, 2 + ix] == 'female', 2 + ix] = 0
    
    # embark c=0, s=1, q=2
    data[data[0::, 9 + ix] == 'C', 9 + ix] = 0
    data[data[0::, 9 + ix] == 'S', 9 + ix] = 1
    data[data[0::, 9 + ix] == 'Q', 9 + ix] = 2
    
    # All the ages with no data make the median of the data
    data[data[0::, 3 + ix] == '', 3 + ix] = np.median(data[data[0::, 3 + ix] != '', 3 + ix].astype(np.float))
    # All missing ebmbarks just make them embark from most common place
    data[data[0::, 9 + ix] == '', 9 + ix] = np.round(np.mean(data[data[0::, 9 + ix] != '', 9 + ix].astype(np.float)))
    
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
    
def load_age_data():
    if os.path.exists("train_non_missing_age_one_hot.pkl"):
        train1 = joblib.load("train_non_missing_age_one_hot.pkl")
        #np.savetxt("train_preprocessed.csv", train, delimiter=",", fmt='%.4f')
    else:
        train1 = pd.read_csv("data/train_non_missing_age_one_hot.csv")
        train1 = np.array(train1)
        #train = process_data(train)
        # train = np.delete(train, [2, 7, 9], 1)  # remove these columns
        joblib.dump(train1, "train_non_missing_age_one_hot.pkl")
        
    if os.path.exists("test_non_missing_age_one_hot.pkl"):
        train2 = joblib.load("test_non_missing_age_one_hot.pkl")
        #np.savetxt("train_preprocessed.csv", train, delimiter=",", fmt='%.4f')
    else:
        train2 = pd.read_csv("data/test_non_missing_age_one_hot.csv")
        train2 = np.array(train2)
        #train = process_data(train)
        # train = np.delete(train, [2, 7, 9], 1)  # remove these columns
        joblib.dump(train2, "test_non_missing_age_one_hot.pkl")
        
    if os.path.exists("train_missing_age_one_hot.pkl"):
        train_miss = joblib.load("train_missing_age_one_hot.pkl")
        np.savetxt("train_missing_age_one_hot.csv", train_miss, delimiter=",", fmt='%.4f')
    else:
        train_miss = pd.read_csv("data/train_missing_age_one_hot.csv")
        train_miss = np.array(train_miss)
        #test = process_data(test, False)
        # test = np.delete(test, [1, 6, 8], 1)  # remove these columns
        joblib.dump(train_miss, "train_miss.pkl")
        
    if os.path.exists("test_missing_age_one_hot.pkl"):
        test_miss = joblib.load("test_missing_age_one_hot.pkl")
        np.savetxt("test_missing_age_one_hot.csv", test_miss, delimiter=",", fmt='%.4f')
    else:
        test_miss = pd.read_csv("data/test_missing_age_one_hot.csv")
        test_miss = np.array(test_miss)
        #test = process_data(test, False)
        # test = np.delete(test, [1, 6, 8], 1)  # remove these columns
        joblib.dump(test_miss, "test_miss.pkl")
        
    #train_Y = np.asarray([x[0] for x in train1] + [x[0] for x in train2])
    #train_X = np.asarray([x[1:] for x in train1] + [x[1:] for x in train2])
    
    train_Y = np.asarray([x[0] for x in train1])
    train_X = np.asarray([x[1:] for x in train1])

    return train_X, train_Y, train_miss, test_miss