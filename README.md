# README

Brief Description of the organisation of the project.
1. Each task is done in Jupyter Notebook using .ipynb file. There are one .ipynb file in each folder A1,A2,A3,A4. Data is extracted, trained and tested with atleast two different models. Each model goes through validation and comparison is made. From this, the best classifier is identified. The .ipynb file shows each task progress basically.

2. With all the required functions identified, two utility .py files are created name taskA_util.py and taskB_util.py. This utility files are created to extract the inputs for each task as well as provide the relevant functions that need to be called later in main.py

3. main.py imports both utility files to be used for the data training and testing. The output of main.py produces training accuracy and testing accuracy for each tasks.

4. dataset folder contains raw image. Meanwhile Datasets folder contains processed image or extracted inputs from image.

5. Some support files are also included in the same directory as main.py(ie. shape_predictor_68_face_landmarks.dat)


Role of each file
1. Main.py - Main python file. Assessor should run this.
2. taskA_util - Utility file for Task A1 and A2.
3. taskB_util - Utility file for Task B1 and B2.
4. A1.ipynb - Progress file for task A1
5. A2.ipynb - Progress file for task A2
6. B1.ipynb - Progress file for task B1
7. B2.ipynb - Progress file for task B2
8. shape_predictor_68_face_landmarks.dat - Landmark file to extract 68 features from image
9. haarcascade_frontalface_default.xml - Landmark file for face detection using Haar cascade
10. haarcascade_smile.xml - Landmark file for smile detection using Haar cascade
11. haarcascade_eye_tree_eyeglasses.xml = Landmark file for eye and glasses detection using Haar cascade

OS Version
Windows 10.0.17763

Python Version
Python 3.6.9 in Conda environment

Packages required
os - Version Unknown
cv2 - Version 4.1.2
dlib - Version 19.7.0
pickle - Version 4.0
requests - Version 2.22.0
numpy - Version 1.17.4
pandas - Version 0.25.3
keras - Version 2.24
sklearn - Version 0.22
matplotlib - Version 3.1.2
