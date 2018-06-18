import os
import numpy as np
from PIL.Image import core as image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import sys
from numpy import set_printoptions
from keras.applications.vgg16 import VGG16
from operator import itemgetter
import pandas as pd
from pickle import load
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix,mean_absolute_error, classification_report, mean_squared_error




# Loading Model
my_model = load_model(filepath='mlpAmazon200eD05.h5')

#We print our model summary
print(my_model.summary())
set_printoptions(precision=4, suppress=True)

# Parameters: Weights and Biases
print('MLP last layer bias:')
print(my_model.get_weights()[-1])
print('MLP last layer weights:')
print(my_model.get_weights()[-2])

DATA_FILE = '../DEAD/y_test.pickle'
df = pd.read_pickle(DATA_FILE)
tags = df
def load_sentences(filename):
	return load(open(filename, 'rb'))
# preprocess
def labels(tags):
    le = LabelEncoder()
    tags = le.fit_transform(tags)
    categories = np.unique(tags)
    categoriess = len(np.unique(tags))
    y = np_utils.to_categorical(tags, categoriess)
    return y,categories

test = load_sentences('../DEAD/tf_test.pickle')
y,cat = labels(tags)
score =my_model.evaluate(test, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
y_pred = my_model.predict(test)
t = cat*y_pred
last = t.sum(axis=1)
#with open('last.pickle', 'wb') as handle:
#		pickle.dump(last, handle, protocol=pickle.HIGHEST_PROTOCOL)
y_test_class = np.argmax(y, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
mean_last = np.mean(last)
mean_y = np.mean(y_test_class)
llast = len(last)
baseline  = np.tile(mean_y, llast)
base_mae = mean_absolute_error(last, baseline)
base_mse = mean_squared_error(last, baseline)
mae=mean_absolute_error(last,y_test_class)
mse=mean_squared_error(last,y_test_class)
print('\n Mean of predictions: {} \n Mean of y: {} \n Mean absolute error: {} \n Mean squared error: {} \n Mean of absolute error baseline: {} \n Mean of squared error baseline: {} '.format(mean_last,mean_y,mae,mse,base_mae,base_mse))
#print(last, file=open('last.txt', 'w'))
#confusion_mlp=confusion_matrix(y_test_class, y_pred_class)
#print(confusion_mlp)
