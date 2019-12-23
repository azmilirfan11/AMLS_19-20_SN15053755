import os
import cv2
import dlib
import pickle

from sklearn.svm import SVC
from taskA_util import split_data

infile1 = open('A_input', 'rb')
featureA = pickle.load(infile1)
infile1.close()

infile2 = open('A_output', 'rb')
labelA = pickle.load(infile2)
infile2.close()


train_img_A1, train_lab_A1, test_img_A1, test_lab_A1 = split_data(featureA, labelA[::2])

# model_A1 = SVC(C=0.01, kernel='Linear', gamma=0.001)
# acc_A1_train = model_A1.fit(train_img_A1, train_lab_A1)
# acc_A1_test = model_A1.predict(test_img_A1, test_lab_A1)

print("\n Process done")
print("\n")
print(len(train_img_A1))

print("\n")
print(len(train_lab_A1))

print("\n")
print(len(test_img_A1))

print("\n")
print(len(test_lab_A1))