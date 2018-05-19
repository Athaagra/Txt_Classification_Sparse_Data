# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:49:13 2018

@author: Kel3vra
"""

import pickle


file = open("Tf-IDF.pickle",'rb')
object_file = pickle.load(file)

print(object_file)

