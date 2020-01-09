# README

Brief Description of the organisation of the project.
1. Each task is done in Jupyter Notebook using .ipynb file. There are one .ipynb file in each folder A1,A2,A3,A4. Data is extracted, trained and tested with atleast two different models. Each model goes through validation and comparison is made. From this, the best classifier is identified. The .ipynb file shows each task progress.

2. With all the required functions identified, two utility .py files are created name taskA_util.py and taskB_util.py. This utility files are created to extract the inputs for each task as well as provide the relevant functions that need to be called later in main.py

3. main.py imports both utility files to be used for the data training and testing. The output of main.py produces training accuracy and testing accuracy for each tasks.

4. dataset folder contains raw image.

5. Datasets folder contains processed/cropped image or extracted inputs from image.

5. Some support files are also included in the same directory as base directory (ie. shape_predictor_68_face_landmarks.dat)

6. New utility file from new test images named new_test_util.py as a support for new test dataset given.


Role of each file
1. main.py - Main python file. Assessor should run this. (need files especially in Datasets folder)
2. taskA_util - Utility file for Task A1 and A2.
3. taskB_util - Utility file for Task B1 and B2.
4. A1.ipynb - Progress file for task A1
5. A2.ipynb - Progress file for task A2
6. B1.ipynb - Progress file for task B1
7. B2.ipynb - Progress file for task B2
8. ./Datasets/A1_feat.dat - Extracted Features for A1
9. ./Datasets/A1_label.dat - Extracted Labels for A1
10. ./Datasets/A2_feat.dat - Extracted Features for A2
11. ./Datasets/A2_label.dat - Extracted Labels for A2
12. ./Datasets/B_df.pkl - Pandas DataFrame for Task B
13. shape_predictor_68_face_landmarks.dat - Landmark file to extract 68 features from image
14. haarcascade_frontalface_default.xml - Landmark file for face detection using Haar cascade
15. haarcascade_smile.xml - Landmark file for smile detection using Haar cascade
16. haarcascade_eye_tree_eyeglasses.xml = Landmark file for eye and glasses detection using Haar cascade
17. dataset_test_AMLS_19-20 - New test dataset given for more testing to be done.

OS Version : Windows 10.0.17763

Python Version : 3.6.9 in Conda environment

Packages required
1. os - Version Unknown
2. cv2 - Version 4.1.2
3. dlib - Version 19.7.0
4. pickle - Version 4.0
5. requests - Version 2.22.0
6. numpy - Version 1.17.4
7. pandas - Version 0.25.3
8. keras - Version 2.24
9. sklearn - Version 0.22
10. matplotlib - Version 3.1.2
