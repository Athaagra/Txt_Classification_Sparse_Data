# -*- coding: utf-8 -*-
"""
Created on Wed May 16 01:02:21 2018

@author: Kel3vra
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pickle import load
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "input"]).decode("utf8"))

DATA_FILE = 'sampleAmazon.csv'
df = pd.read_csv(DATA_FILE,encoding='latin-1')
print(df.head())
tags = df.overall
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import time
from keras import metrics
from keras.utils import np_utils
print('import done')

batch_size=32

# preprocess
def labels(tags):
    le = LabelEncoder()
    tags = le.fit_transform(tags)
    categories = len(np.unique(tags))
    y = np_utils.to_categorical(tags, categories)
    return y
y = labels(tags)

# load a  dataset
def load_sentences(filename):
	return load(open(filename, 'rb'))
# our model
def get_simple_model(num_max):
   model = Sequential()
   model.add(Dense(512, input_shape=(num_max,)))
   model.add(Activation('relu'))
   model.add(Dropout(0.2))
   model.add(Dense(256))
   model.add(Activation('relu'))
   model.add(Dropout(0.2))
   model.add(Dense(5)) #Last layer with one output per class
   model.add(Activation('softmax'))
   model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
   print('compile done')
   return model
def sparse(dataset):
    x  = csr_matrix(dataset)
   # X = X_train_sparse.todense()
   # x = np.array(X)
    num_max = x.shape[1]
    return x,num_max

def splitting(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    X_train, x_cross, y_train, y_cross = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    return X_train,y_train,X_test,y_test,x_cross,y_cross

def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = X.shape[0]/batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def check_model(model,x,y,x_cross,y_cross):
   model.fit_generator(generator=batch_generator(x, y, 10, True),
                       epochs=50,
                       validation_data=(x_cross.todense(), y_cross),
                       samples_per_epoch=100)
   model.save(filepath=r'mlpAmazon.h5', overwrite=True)


dataset = load_sentences('Tf-IDF.pickle')
x,num_max = sparse(dataset)
x_train,y_train,x_test,y_test,x_cross,y_cross = splitting(x,y)
m = get_simple_model(num_max)
check_model(m,x_train,y_train,x_cross,y_cross)