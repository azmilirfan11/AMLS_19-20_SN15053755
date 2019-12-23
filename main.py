# ======================================================================================================================
# Importing library
import os
import cv2
import dlib
import pickle

from sklearn.svm import SVC
from taskA_util import split_data
from tasA_util import A1
# ======================================================================================================================
# Importing data
infile1 = open('A_input', 'rb')
featureA = pickle.load(infile1)
infile1.close()

infile2 = open('A_output', 'rb')
labelA = pickle.load(infile2)
infile2.close()

infile3 = open('B_input', 'rb')
featureB = pickle.load(infile3)
infile3.close()

infile4 = open('B_output', 'rb')
labelB = pickle.load(infile4)
infile4.close()

# ======================================================================================================================
# Data preprocessing
# TaskA1
train_img_A1, train_lab_A1, test_img_A1, test_lab_A1 = split_data(featureA, labelA[::2])
# TaskA2
train_img_A2, train_lab_A2, test_img_A2, test_lab_A2 = split_data(featureA, labelA[1::2])

# TaskB1
data_train, data_val, data_test = data_preprocessing(args...)

# ======================================================================================================================
# Task A1
model_A1 = A1(C=0.01, kernel='Linear', gamma=0.001)
acc_A1_train = model_A1.train(train_img_A1, train_lab_A1)
acc_A1_test = model_A1.test(test_img_A1, test_lab_A1)

# ======================================================================================================================
# Task A2
model_A2 = A2(args...)
acc_A2_train = model_A2.train(args...)
acc_A2_test = model_A2.test(args...)
Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
model_B1 = B1(args...)
acc_B1_train = model_B1.train(args...)
acc_B1_test = model_B1.test(args...)
Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
model_B2 = B2(args...)
acc_B2_train = model_B2.train(args...)
acc_B2_test = model_B2.test(args...)
Clean up memory/GPU etc...



# ======================================================================================================================
# Task A1
model_A1 = A1(args...)                 # Build model object.
acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
Clean up memory/GPU etc...             # Some code to free memory if necessary.





# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'