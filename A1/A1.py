import os
import cv2
import dlib
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.preprocessing import image
from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

# os.chdir("..")

infile_feat = open('A_input', 'rb')
feature = pickle.load(infile_feat)
infile_feat.close()

infile_label = open('A_output', 'rb')
alllabel = pickle.load(infile_label)
infile_label.close()

gender = alllabel[::2]
#smiling = alllabel[1::2]



