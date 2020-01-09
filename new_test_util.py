# ======================================================================================================================
# Import Library
import os
import cv2
import dlib
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# ======================================================================================================================
# Directory List and Filename
labels_filename = 'labels.csv'

basedir = os.path.abspath(os.curdir)
test_dir = os.path.join(basedir,'dataset_test_AMLS_19-20')
celeba_dir = os.path.join(test_dir,'celeba_test')
cartoon_dir = os.path.join(test_dir,'cartoon_set_test')

celeb_img_dir = os.path.join(celeba_dir,'img')
celeb_lab_dir = os.path.join(celeba_dir,labels_filename)
cart_img_dir = os.path.join(cartoon_dir,'img')
cart_lab_dir = os.path.join(cartoon_dir,labels_filename)

# New directory for cropped
crop_img_dir = os.path.join(test_dir,'crop_test')

# ======================================================================================================================
# Main Function
def main():
    ## Checker so that not multiple times download
    downloadcheck = True
    ## Cut images for B2 task and save to crop_test folder
    print("Cropping Images\n")
    print("This might takes a while...\n")
    if downloadcheck==False:
        cropimage(cart_img_dir, crop_img_dir)
        downloadcheck=True
    print("Cropping successful and new images saved to crop_test folder\n")
    print("\n")

    print("Getting data...\n") # just for debugging purpose
    test_featA1, test_labA1, test_featA2, test_labA2, test_B1_gen, test_B2_gen = get_test_data(celeb_img_dir, celeb_lab_dir, cart_img_dir, cart_lab_dir)


# ======================================================================================================================
# Support Functions
def extract_features_labels(images_dir, labels_dir, select):
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


def createdataframe(labels_dir):
    labels_file = open(labels_dir, 'r')
    lines = labels_file.readlines()
    eye_label = {line.split('\t')[3].split('\n')[0] : int(line.split('\t')[1])+1 for line in lines[1:]}

    fshape =[]
    for line in lines[1:]:
        temp = str(int(line.split('\t')[2])+1)
        fshape.append(temp)

    df = pd.DataFrame(list(eye_label.items()))
    df.columns = ['FileName', 'EyeLabel']


    EyeColour = []
    for eye in df.EyeLabel:
        if eye == 1:
            EyeColour.append("Brown")
        elif eye == 2:
            EyeColour.append("Blue")
        elif eye == 3:
            EyeColour.append("Green")
        elif eye == 4:
            EyeColour.append("Gray")
        else:
            EyeColour.append("Black")

    df['EyeColour'] = EyeColour
    df['FaceShape'] = fshape

    return df


def cropimage(images_dir, new_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    if os.path.isdir(images_dir):
        for img_path in image_paths:
            file_name = img_path.split('\\')[-1] ## Filename include .png
            img = cv2.imread(img_path)
            crop = img[230:230+58, 175:175+58]
            cv2.imwrite(os.path.join(new_dir, file_name), crop)


# ======================================================================================================================
# Functions to be called in main.py
def get_test_data(celeb_img_dir, celeb_lab_dir, cart_img_dir, cart_lab_dir):
    # For A1
    test_featA1, test_labA1, _ = extract_features_labels(celeb_img_dir, celeb_lab_dir, select=1)
    test_featA1 = test_featA1.reshape((len(test_featA1), 68*2))
    test_labA1 = totuple(test_labA1)

    # For A2
    test_featA2, test_labA2, _ = extract_features_labels(celeb_img_dir, celeb_lab_dir, select=2)
    test_featA2 = test_featA2.reshape((len(test_featA2), 37*2))
    test_labA2 = totuple(test_labA2)

    # For B
    test_dataframe = createdataframe(cart_lab_dir)
    test_datagen = ImageDataGenerator(rescale=1./255)
    # For B1
    test_B1_gen = test_datagen.flow_from_dataframe(test_dataframe, cart_img_dir, x_col="FileName", y_col="FaceShape",
                                            class_mode="categorical", target_size=(32,32), batch_size=1, shuffle=False)

    # For B2
    test_B2_gen = test_datagen.flow_from_dataframe(test_dataframe, crop_img_dir, x_col="FileName", y_col="EyeColour",
                                            class_mode="categorical", target_size=(32,32), batch_size=1, shuffle=False)


    return test_featA1, test_labA1, test_featA2, test_labA2, test_B1_gen, test_B2_gen


# ======================================================================================================================
# Run Main Function
if __name__ == "__main__":
    main()