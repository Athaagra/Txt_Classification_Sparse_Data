# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:44:29 2018

@author: Kel3vra
"""
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys
import json
import pickle
from tempfile import TemporaryFile
outfile = TemporaryFile()

def tokenization(x):
    return x.split(" ")

def CountVect(x, y, num_features = 5000):
	"""
    Function for creating a matrix of Frequencies (counts), and then splitting in training and test sets.
    Inputs: x = list of reviews, as returned from review_to_words()
            y = column with overall rating
            num_features = Number of columns for the Tf-Idf matrix
    Output: Training and test sets containing frequencies
	"""
	vectorizer = CountVectorizer(analyzer = "word",   \
                                 token_pattern = "\w*[a-z]\w*",\
                                 preprocessor = None, \
                                 min_df = 10,\
                                 stop_words = 'english',   \
                                 max_features = num_features)

	train_data_features = vectorizer.fit_transform(x)
	vocab = vectorizer.get_feature_names()
	with open('CountVector.pickle', 'wb') as handle:
		pickle.dump(train_data_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
	return train_data_features

if __name__=='__main__':
	data = pd.read_csv("../item_dedup2.csv",keep_default_na=False)
	print('\nData Loaded.')
	review = np.array(data['reviewText'])
	overall = np.array(data['overall'])
	start_time = time.time()
	train_data_features = CountVect(review, overall)
	elapsed_time = time.time() - start_time
	print('microsecond:',elapsed_time,file=open('CountVect_times.txt', 'a'))
	elapsed_time2 = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
	print('24-hours:',elapsed_time2,file=open('CountVect_times.txt', 'a'))
