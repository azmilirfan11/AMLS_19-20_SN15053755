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

# %matplotlib inline
# # required magic function

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

labels_filename = 'labels.csv'
feat1_sav = 'A1_feat.dat'
lab1_sav = 'A1_label.dat'
feat2_sav = 'A2_feat.dat'
lab2_sav = 'A2_label.dat'
## File is saved in output for both gender and smiling
## Odd is gender file, Even is smiling file

basedir = os.path.abspath(os.curdir)
dataset_dir = os.path.join(basedir,'dataset')
celeba_dir = os.path.join(dataset_dir,'celeba')
images_dir = os.path.join(celeba_dir,'img')
labels_dir = os.path.join(celeba_dir,labels_filename)

def main():
    featureA1, labelA1,_ = extract_features_labels(images_dir, labels_dir, select=1)
    
    os.chdir("./Datasets")
    ## saving features extraction data
    savfile_feat1 = open(feat1_sav, 'wb')
    pickle.dump(featureA1, savfile_feat1)
    savfile_feat1.close()

    ## saving labels data
    savfile_label1 = open(lab1_sav, 'wb')
    pickle.dump(labelA1, savfile_label1)
    savfile_label1.close()

    featureA2, labelA2, _ = extract_features_labels(images_dir, labels_dir, select=2)

    savfile_feat2 = open(feat2_sav, 'wb')
    pickle.dump(featureA2, savfile_feat2)
    savfile_feat2.close()
    
    savfile_label2 = open(lab2_sav, 'wb')
    pickle.dump(labelA2, savfile_label2)
    savfile_label2.close()

    os.chdir("..")


def extract_features_labels(images_dir, labels_dir, select=1):
    """ return:
        landmark_features:  an array containing 68 landmark points for each image in celeba folder
        labels:  an array containing (select = 1 , gender label) (select = 2 , smiling label) (select = None, both)

    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)] ##filename in matrix
    target_size = None
    labels_file = open(labels_dir, 'r')
    lines = labels_file.readlines()
    lab_gen = {line.split('\t')[0] : int(line.split('\t')[2]) for line in lines[1:]} 
    lab_smi = {line.split('\t')[0] : int(line.split('\t')[3]) for line in lines[1:]}

        
    if os.path.isdir(images_dir):
        all_features = []
        reduced_features = []
        gender_labels = []
        smiling_labels = []
        error_features = []

        for img_path in image_paths:
            file_name= img_path.split('.')[0].split('\\')[-1] ##getting name of file; remove png/jpg + dir

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            x = features
            if features is not None:
                all_features.append(features)
                reduced_features.append(np.concatenate((x[0:17],x[48:68])))
                gender_labels.append(lab_gen[file_name])
                smiling_labels.append(lab_smi[file_name])
            if features is None:
                error_features.append(file_name)
                

    landmark_features = np.array(all_features)
    reduce = np.array(reduced_features)
    gender_label = (np.array(gender_labels) +1)/2
    smiling_label = (np.array(smiling_labels) + 1)/2 # converts the -1 into 0, so male=0 and female=1

    if select == 1:
        return landmark_features, gender_label, error_features
    elif select == 2:
        return reduce, smiling_label, error_features
    else:
        return -1

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image
    
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def split_data_68(image_feature, image_label):
    i,j,k,l = train_test_split(image_feature, image_label, test_size = 0.2, random_state=42)
    train_image = i.reshape((len(i), 68*2))
    train_label = totuple(k)
    test_image = j.reshape((len(j), 68*2))
    test_label = totuple(l)
    
    return train_image, train_label, test_image, test_label

def split_data_37(image_feature, image_label):
    i,j,k,l = train_test_split(image_feature, image_label, test_size = 0.2, random_state=42)
    train_image = i.reshape((len(i), 37*2))
    train_label = totuple(k)
    test_image = j.reshape((len(j), 37*2))
    test_label = totuple(l)
    
    return train_image, train_label, test_image, test_label

# class A1(BaseEstimator, ClassifierMixin):
#     def __init__(self):
#         SVC(C=Cs, gamma=gammas, kernel=kernels)

#     def train(self, trainingdata, traininglabel):
#         A1.fit(trainingdata, traininglabel)
#         ## crossvalidation here
#         # return crossvalidation score



if __name__ == "__main__":
    main()