import math
import cv2
import time
from cv2 import imshow
from cv2 import THRESH_BINARY
import matplotlib.pyplot as plt
import numpy as np

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
    # crop
    
    height, width, _ = frame.shape
    #frame = frame[round(height / 1.4):height, round(width / 4):round(width / 4) * 3, :]
    fps = cap.get(5)
    gray_frame = frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2]

    #DENOISE
    #the first parameter is the source image to denoise, the second parameter indicates the size of the filter, the last parameter means
    # 0 = constant border, it is used to specify the border type of the image
    gray_frame = cv2.GaussianBlur(gray_frame,(5,5),0)


    #viewer.clear()
    #viewer.hist(gray_frame.flatten(), bins=255)
    #fig.canvas.draw()
    #plt.pause(0.08)

    if valid:
        th,edge = cv2.threshold(gray_frame,0,255,cv2.THRESH_OTSU)
       
        cv2.imshow('original', frame)
        cv2.imshow('edge', edge)
        cv2.waitKey(5)
    else:
        break
cap.release()
cv2.destroyAllWindows()
