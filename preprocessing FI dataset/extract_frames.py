import cv2
import math
import sys

videoName = sys.argv[1]

cap = cv2.VideoCapture('DATA/' + videoName)
frameRate = cap.get(5) #frame rate
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = "OUTPUT/" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
cap.release()
