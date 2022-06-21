import math
import cv2
import time
from cv2 import imshow
import matplotlib.pyplot as plt
import numpy as np



print("OpenCV version is: " + cv2.__version__)

#video acquisition, cap means captured
cap = cv2.VideoCapture("D:\Road_Videos\\Virb0462.m4v")

# img = cv2.imread("test_data\\first_example.png")


if not cap.isOpened():
    print("Could not open video file")

#window definition in order to show the video 
cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.namedWindow("edge", cv2.WINDOW_NORMAL)

#HISTOGRAM initialization part
fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion()  # Turns interactive mode on (probably unnecessary)
#fig.show()


while cap.isOpened():
    #read method: grabs, decodes and return the next video frame. The image frame is saved in "frame". The return value (false as an example
    # if no image is captured) is saved in "valid".
    #https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
    
    valid, frame = cap.read()

    height, width, _ = frame.shape

    #crop
    
    #frame = frame[round(height / 1.4):height, round(width / 4):round(width / 4) * 3, :]

    #Returns the specified videocapture property
    #https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    fps = cap.get(5)  # 5 = CAP_PROP_FPS 

    #obtains a gray frame by summing all the channel layers of the frame
    gray_frame = frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2]

    #HISTOGRAM PART
    #viewer.clear()
    #viewer.hist(gray_frame.flatten(), bins=255)
    #fig.canvas.draw()
    #plt.pause(0.08)

    if valid:
        edge = cv2.Canny(frame, 180, 255)
        cv2.imshow('original', frame)
        cv2.imshow('edge', edge)
        #display image until a key is pressed or until 5 milliseconds pass
        cv2.waitKey(5)
    else:
        break

#closes the video file
cap.release()
#clear all the opened windows
cv2.destroyAllWindows()

"""
Questo mi è piaciuto, considerando che ho usato solo
operazioni fra colori
cap = cv2.VideoCapture("test_data\\Long Voyage.mp4")
#img = cv2.imread("test_data\\first_example.png")
if not cap.isOpened():
    print("Could not open video file")
cv2.namedWindow("original", cv2.WINDOW_NORMAL) 
cv2.namedWindow("edge", cv2.WINDOW_NORMAL) 
fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion() # Turns interactive mode on (probably unnecessary)
fig.show()
# Gray colors have almost the same value in all channels
while cap.isOpened():
    valid, frame = cap.read()
    # crop
    height, width, _ = frame.shape
    frame = frame[round(height/1.4):height, round(width/4):round(width/4)*3, :]


    frame0 = frame[:, :, 0]
    frame1 = frame[:, :, 1]
    frame2 = frame[:, :, 2]
    frame0 = np.where(
        (np.abs(frame0 - frame1) < 200) &
        (np.abs(frame0 - frame2) < 200)
        , frame0, 255)
    frame1 = np.where(
        (np.abs(frame1 - frame0) < 200) &
        (np.abs(frame1 - frame2) < 200)
        , frame1, 255)
    frame2 = np.where(
        (np.abs(frame2 - frame0) < 200) &
        (np.abs(frame2 - frame1) < 200)
        , frame2, 255)     
    frame = np.stack((frame0, frame1, frame2), axis=-1)
    print(frame.shape)
    gray_frame = frame[: , :, 0] + frame[: , :, 1] + frame[: , :, 2]
    viewer.clear()
    viewer.hist(gray_frame.flatten(), bins=255)
    fig.canvas.draw()
    plt.pause(0.08)
    if valid:
        #edge = cv2.Canny(uncolour, 180, 255)
        cv2.imshow('original', frame)
        #cv2.imshow('edge', edge)
        cv2.waitKey(20)
    else:
        break
cap.release()
cv2.destroyAllWindows()
"""
