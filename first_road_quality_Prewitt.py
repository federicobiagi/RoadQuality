import math
import cv2
import time
from cv2 import imshow
from cv2 import THRESH_BINARY
import matplotlib.pyplot as plt
import numpy as np

print("OpenCV version is: " + cv2.__version__)

def PreWitt_Mask3x3(frame):
    #Prewitt Masks
    Kx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    Ky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    #Apply the filter on the gray scale image, the -1 value mantains the same depth as the original image
    filteredX = cv2.filter2D(frame,-1,Kx)
    filteredY = cv2.filter2D(frame,-1,Ky)

    filtered = filteredX + filteredY    
    return filtered




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
       
        edge = PreWitt_Mask3x3(gray_frame)
        cv2.imshow('original', frame)
        cv2.imshow('edge', edge)
        cv2.waitKey(5)
    else:
        break
cap.release()
cv2.destroyAllWindows()
