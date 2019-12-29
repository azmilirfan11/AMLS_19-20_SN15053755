# ======================================================================================================================
# Importing library
import os
import cv2
import dlib
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
## From utility file
from taskA_util import split_data_68, split_data_37
from taskA_util import Atrain as train_A
from taskA_util import Atest as test_A
from taskB_util import Btrain as train_B
from taskB_util import Btest as test_B
from taskB_util import createmodel

# ======================================================================================================================
#  Label filename
labels_filename = 'labels.csv'
# Directory List
basedir = os.path.abspath(os.curdir)
dataset_dir = os.path.join(basedir,'dataset')
# For celeba
celeba_dir = os.path.join(dataset_dir,'celeba')
celeb_img_dir = os.path.join(celeba_dir, 'img')
celeb_lab_dir = os.path.join(celeba_dir, labels_filename)
# For cartoon_set
cartoon_dir = os.path.join(dataset_dir,'cartoon_set')
cart_img_dir = os.path.join(cartoon_dir,'img')
cart_lab_dir = os.path.join(cartoon_dir, labels_filename)
# Directory for preprocessed data
prepro_dir = os.path.join(basedir, 'Datasets')

# ======================================================================================================================
# Importing Data
# Task A1
# Feature A1
infile11 = open(os.path.join(prepro_dir,'A1_feat.dat'), 'rb')
featureA1 = pickle.load(infile11)
infile11.close()
# Label A1
infile12 = open(os.path.join(prepro_dir,'A1_label.dat'), 'rb')
labelA1 = pickle.load(infile12)
infile12.close()
#
# Task A2
# Feature A2
infile21 = open(os.path.join(prepro_dir,'A2_feat.dat'), 'rb')
featureA2 = pickle.load(infile21)
infile21.close()
# Label A2
infile22 = open(os.path.join(prepro_dir,'A2_label.dat'), 'rb')
labelA2 = pickle.load(infile22)
infile22.close()
#
# Task B DataFrame
B_df = pd.read_pickle(os.path.join(prepro_dir, 'B_df.pkl'))
# Task B1
# modelB1_json = os.path.join(prepro_dir, 'modelB1_CNN.json')
datagenB1 = ImageDataGenerator(rescale=1./255., validation_split=0.25, horizontal_flip=True, vertical_flip=True)
testdatagenB1 = ImageDataGenerator(rescale=1./255)
#
# Task B2
# modelB2_json = os.path.join(prepro_dir, 'modelB2_CNN.json')
datagenB2 = ImageDataGenerator(rescale=1./255., validation_split=0.25, horizontal_flip=True, vertical_flip=True)
testdatagenB2 = ImageDataGenerator(rescale=1./255)

# ======================================================================================================================
# Data Preprocessing
# TaskA1
train_img_A1, train_lab_A1, test_img_A1, test_lab_A1 = split_data_68(featureA1, labelA1)

# TaskA2
train_img_A2, train_lab_A2, test_img_A2, test_lab_A2 = split_data_37(featureA2, labelA2)

# TaskB1
trainB1_df, testB1_df = train_test_split(B_df, random_state=99)
train_B1_gen = datagenB1.flow_from_dataframe(trainB1_df, cart_img_dir, x_col="FileName", y_col="FaceShape",
                                            class_mode="categorical", target_size=(32,32), batch_size=32, subset='training')
val_B1_gen = datagenB1.flow_from_dataframe(trainB1_df, cart_img_dir, x_col="FileName", y_col="FaceShape",
                                            class_mode="categorical", target_size=(32,32), batch_size=32, subset='validation')
test_B1_gen = testdatagenB1.flow_from_dataframe(testB1_df, cart_img_dir, x_col="FileName", y_col="FaceShape",
                                            class_mode="categorical", target_size=(32,32), batch_size=1, shuffle=False)

# TaskB2
trainB2_df, testB2_df = train_test_split(B_df, random_state=11)
train_B2_gen = datagenB2.flow_from_dataframe(trainB2_df, prepro_dir, x_col="FileName", y_col="EyeColour",
                                            class_mode="categorical", target_size=(32,32), batch_size=32, subset='training')
val_B2_gen = datagenB2.flow_from_dataframe(trainB2_df, prepro_dir, x_col="FileName", y_col="EyeColour",
                                            class_mode="categorical", target_size=(32,32), batch_size=32, subset='validation')
test_B2_gen = testdatagenB2.flow_from_dataframe(testB2_df, prepro_dir, x_col="FileName", y_col="EyeColour",
                                            class_mode="categorical", target_size=(32,32), batch_size=1, shuffle=False)

# ======================================================================================================================
# Task A1
model_A1 = SVC(C=0.01, kernel='linear', gamma=0.001)
model_A1, acc_A1_train = train_A(model_A1, train_img_A1, train_lab_A1)
acc_A1_test = test_A(model_A1, test_img_A1, test_lab_A1)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task A2
model_A2 = SVC(C=0.1, kernel='linear', gamma=0.001)
model_A2, acc_A2_train = train_A(model_A2, train_img_A2, train_lab_A2)
acc_A2_test = test_A(model_A2, test_img_A2, test_lab_A2)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
# with open(modelB1_json, 'r') as B1_json:
#     model_B1 = model_from_json(B1_json.read())
model_B1 = createmodel()
acc_B1_train = train_B(model_B1, train_B1_gen, val_B1_gen, num_epoch = 16)
acc_B1_test = test_B(model_B1, test_B1_gen)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
model_B2 = createmodel()
acc_B2_train = train_B(model_B2, train_B2_gen, val_B2_gen, num_epoch = 11)
acc_B2_test = test_B(model_B2, test_B2_gen)
# Clean up memory/GPU etc...



# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'