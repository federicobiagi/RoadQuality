import math
import cv2
import time
from cv2 import imshow
from cv2 import THRESH_BINARY
import matplotlib.pyplot as plt
import numpy as np
import torch

print("OpenCV version is: " + cv2.__version__)


cap = cv2.VideoCapture("D:\Road_Videos\\Road2.mp4")

# img = cv2.imread("test_data\\first_example.png")

if not cap.isOpened():
    print("Could not open video file")


cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.namedWindow("edge", cv2.WINDOW_NORMAL)

fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion()  # Turns interactive mode on (probably unnecessary)
#fig.show()


while cap.isOpened():
    valid, frame = cap.read()
    #print(frame.dtype)
    #frame = np.clip(frame,0,255)
    height, width, _ = frame.shape
    #frame = frame[round(height / 1.4):height, round(width / 4):round(width / 4) * 3, :]
    fps = cap.get(5)
    #to produce the gray frame we sum along the channels thus eliminating the third dimension, the result is a gray scale image with infact no channel
    gray_frame = frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2]
    
    #normalization
    #frame = frame.astype("float32")
    #grayframe = frame[:,:,0] + frame[:,:,1] + frame[:,:,2]
    #grayframe = grayframe / 3
    #grayframe = grayframe.astype("uint8")
   
    #DENOISE
    #the first parameter is the source image to denoise, the second parameter indicates the size of the filter, the last parameter means
    # 0 = constant border, it is used to specify the border type of the image
    #gray_frame = cv2.GaussianBlur(gray_frame,(5,5),0)


    viewer.clear()
    

    if valid:
        #the first parameter needs to be a gray scale image, the second parameter determines the threshold value,
        #the third one is for the maximum value of the see with the threshold, the type is used to set optional 
        #thresholding types liek otsu. The method determines the best threshold to use between the two specified thresholds.
        th,edge = cv2.threshold(gray_frame, thresh= 0, maxval=255,type = cv2.THRESH_OTSU)
       
        

        cv2.imshow('original', frame)
        cv2.imshow('edge', edge)
        cv2.waitKey(5)
    else:
        break
cap.release()
cv2.destroyAllWindows()
