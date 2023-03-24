import cv2
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline


def preprocess(img):
    # Preprocess the given image "img".
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
    # Blur the image to remove any noise in it.
    blurred = cv2.blur(gray,(3,3))  

    # Convert the grayscale image to a binary imageAp by applying a threshold between 50 and 255 on the blurred image.
    # The pixels having values less than 50 will be considered 0, and 255 otherwise.
    _, thresholded_img = cv2.threshold(blurred,50,255,cv2.THRESH_BINARY)
   
    return thresholded_img


# Find the contours and the contours area of a given image (img)
def findContourArea(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[1])
    return area, contours

# Find the minimum bounding rectangle that can fit the given contours
def findBoundingRectangleArea(img, contours):
    x,y,w,h = cv2.boundingRect(contours[1])
    area = w*h    
    bounding_rectangle = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (x, y), (x + w, y + h), (0, 255, 0), 2)
    return area, bounding_rectangle

# Find the minimum enclosing circle that can fit the given contours 
def findBoundingCircleArea(img, contours):   
    (x_axis,y_axis),radius = cv2.minEnclosingCircle(contours[1])
    radius=int(radius)    
    area = math.pi * radius * radius
    center=(int(x_axis),int(y_axis))
    bounding_circle = cv2.circle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), center, int(radius), (0, 255, 0), 2)
    return area, bounding_circle

# Find the minimum enclosing triangle that can fit the given contours
def findBoundingTriangleArea(img, contours):
    x = cv2.minEnclosingTriangle(contours[1])
    area = x[0]
    bounding_triangle = cv2.polylines(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), np.int32([x[1]]), True, (0, 255, 0), 2)
    return area, bounding_triangle


def extract_features(img):
    area, contours = findContourArea(img)
    area1, _ = findBoundingRectangleArea(img, contours)
    area2, _ = findBoundingCircleArea(img, contours)
    area3, _ = findBoundingTriangleArea(img, contours)
    features = []
    features.append(area/area1)
    features.append(area/area2)
    features.append(area/area3)
    return features