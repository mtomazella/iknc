import numpy as np
import cv2 as cv

availableCameras = []
for i in range(10):
    try:
      cap = cv.VideoCapture(i)
      if not cap.isOpened():
          continue
      else:
          availableCameras.append(i)
      cap.release()
    except:
        continue

print(availableCameras)
 
leftCamera = cv.VideoCapture(availableCameras[1])
rightCamera = cv.VideoCapture(availableCameras[2])

if not leftCamera.isOpened():
    print("Cannot open left camera")
    exit()

if not leftCamera.isOpened():
    print("Cannot open right camera")
    exit()

while True:
    leftCameraResponse, leftFrame = leftCamera.read()
    rightCameraResponse, rightFrame = rightCamera.read()
 
    if not leftCameraResponse or not rightCameraResponse:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # gray = cv.cvtColor(leftFrame, cv.COLOR_BGR2GRAY)

    cv.imshow('left', leftFrame)
    cv.imshow('right', rightFrame)

    if cv.waitKey(1) == ord('q'):
        break