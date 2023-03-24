import cv2
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ml_classifier_utilis import *
%matplotlib inline

# Collecting the hand-drawn shapes training set
x_train = [] # list of training images
y_train = [] # list of training labels
shapes = ['Rectangle', 'Circle', 'Triangle']

## Load rectangle images
for filename in sorted(glob.glob('images/rectangle/*.png')):
    img = cv2.imread(filename) ## cv2.imread reads images in RGB format
    x_train.append(img)
    y_train.append(0)

## Load circle images
for filename in sorted(glob.glob('images/circle/*.png')):
    img = cv2.imread(filename) 
    x_train.append(img)
    y_train.append(1)

## Load triangle images
for filename in sorted(glob.glob('images/triangle/*.png')):
    img = cv2.imread(filename)
    x_train.append(img)
    y_train.append(2)

# Converting lists into NumPy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

'''
We have 19 images of hand-drawn shapes, where each image is an RGB image of dimensions 200x200x3. 
(images, height, width, channels)
The size of the training set is:  (19, 200, 200, 3)
The size of the labels of the training set is:  (19,)
'''

# Visualize the dataset
''' 
You can run the following cell multiple times to view different shapes in the
dataset as well as the ground-truth value associated with this training image
'''
# Generate a random index from 0 to 18 inclusive. 
random_index = np.random.randint(0,18)

# Plot the image.
plt.imshow(x_train[random_index])
plt.axis("off")
print("The ground-truth value of this image is: " , shapes[(y_train[random_index])])


# Data preprocessing 
'''
The data is collected from different sources in raw format which is not feasible for the analysis. 
We need to prepare it to be suitable for our problem
'''
# Pre-process each image in the training set 
# Add the pre-processed images to a new list (x_train_preprocessed.)
x_train_preprocessed = []
for i in range(x_train.shape[0]):
    preprocessed_img = preprocess(x_train[i])
    x_train_preprocessed.append(preprocessed_img)
x_train_preprocessed = np.asarray(x_train_preprocessed)
# The size of the training set now is:  (19, 200, 200)


# Visualize what happened after preprocessing 
# Generate a random index from 0 to 18 inclusive. 
random_index = np.random.randint(0,18)

# Plot the image before preprocessing.
plt.imshow(x_train[random_index],cmap='gray')
plt.axis("off")
plt.title("Before Preprocessing")
plt.show()

# Plot the image after preprocessing.
plt.imshow(x_train_preprocessed[random_index], cmap='gray')
plt.axis("off")
plt.title("After Preprocessing")
plt.show()


# Work on the preprocessed training set and drop the raw
x_train = x_train_preprocessed

# Feature extraction
'''
Each image will be represented with a feature vector in three dimensions (x, y and z)
where each component represents the ratio between the area of the figure to the area of 
the bounding rectangle, circle and triangle respectively.
'''

# Create a matrix of zeros to accomodate the training features for all images in the training set
training_features = np.zeros((19,3))

# Populate the array training_features with the extracted features from each image using the above functions
for i in range(training_features.shape[0]):
    training_features[i] = extract_features(x_train[i])




