# ======================================================================================================================
# Import Library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# ======================================================================================================================
# Directory List
labels_filename = 'labels.csv'

basedir = os.path.abspath(os.curdir)
dataset_dir = os.path.join(basedir,'dataset')
cartoon_dir = os.path.join(dataset_dir,'cartoon_set')
images_dir = os.path.join(cartoon_dir,'img')
labels_dir = os.path.join(cartoon_dir, labels_filename)
prepro_dir = os.path.join(basedir,'Datasets')

# ======================================================================================================================
# Save Filename
modelB1_sav = 'modelB1_CNN.json'
modelB2_sav = 'modelB2_CNN.json'
taskB_df = 'B_df.pkl'

# ======================================================================================================================
# Main Function
def main():
    ## Checker so that not multiple times download
    downloadcheck = True
    ## Get Dataframe
    print("Creating Dataframe from CSV file\n")
    dfCNN = createdataframe()

    ## Save Dataframe using pickle
    print("Saving dataframe to Datasets folder\n")
    dfCNN.to_pickle(os.path.join(prepro_dir,taskB_df))
    print("Dataframe saved\n")
    print("\n")

    ## Cut images for B2 task and save to Datasets folder
    print("Cropping Image for TaskB2\n")
    print("This might takes a while...")
    if downloadcheck==False:
        cropimage(images_dir, prepro_dir)
        downloadcheck=True
    print("Cropping successful and new images saved to Datasets folder\n")
    print("\n")

    ### UNUSED ACTUALLY FOR BOTH.. ERROR ""
    ## Create CNN Model and save the raw model
    print("Creating Model for task B1\n")
    modelB1 = createmodel()
    os.chdir("./Datasets")
    with open(modelB1_sav, 'w') as jsonB1:
        jsonB1.write(modelB1.to_json())
    print("Model for B1 is saved to Datasets folder\n")
    print("\n")
    os.chdir("..")

    ## Create CNN Model for B2 this time
    print("Creating Model for task B2\n")
    modelB2 = createmodel()
    os.chdir("./Datasets")
    with open(modelB2_sav, 'w') as jsonB2:
        jsonB2.write(modelB2.to_json())
    print("Model for B2 is saved to Datasets folder\n")
    print("\n")
    os.chdir("..")

    
# ======================================================================================================================
# Support Functions
def cropimage(images_dir, new_dir):
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    if os.path.isdir(images_dir):
        for img_path in image_paths:
            file_name = img_path.split('\\')[-1] ## Filename include .png
            img = cv2.imread(img_path)
            crop = img[230:230+58, 175:175+58]
            cv2.imwrite(os.path.join(new_dir, file_name), crop)

def createdataframe():
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

# ======================================================================================================================
# Functions to be called in main.py
def createmodel():
    model = Sequential()
    
    model.add(Conv2D(24, (3,3), input_shape=(32,32,3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(24, (3,3)))
    model.add(Activation("relu"))

    model.add(Conv2D(48, (3,3)))
    model.add(Activation("relu"))

    model.add(Conv2D(96, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(5))
    model.add(Activation("softmax"))

    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
    
def Btrain(model, traingen, valgen, num_epoch=25, batch_size=32):
    history = model.fit_generator(traingen, steps_per_epoch = traingen.samples // batch_size,
                        validation_data = valgen, validation_steps = valgen.samples // batch_size, epochs = num_epoch)
    validation_accuracy = history.history['val_acc']
    return validation_accuracy[-1]

def Btest(model, testgen):
    filenames = testgen.filenames
    nb_samples = len(filenames)
    prob = model.predict_generator(testgen, steps=nb_samples)
    pred = np.argmax(prob, axis=1)
    true = np.array(testgen.classes)
    acc = accuracy_score(true, pred)
    return acc


# ======================================================================================================================
# Run Main Function
if __name__ == "__main__":
    main()