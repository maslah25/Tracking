import cv2
from Prac import prac

import numpy as np

class Foreground_Background:
    def __init__(self):
        self.roi=((0,800),(15,1280))
        self.objdetect = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)
    def process(self,frame):
        mask = self.objdetect.apply(frame[self.roi[0][0]:self.roi[0][1],self.roi[1][0]:self.roi[1][1]])
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        self.result=mask
    def getResult(self):

        return "Forground_Background", self.result


processors = [Foreground_Background(), prac()]

capture = cv2.VideoCapture("Frame/busystreet.mp4")

while True:
    ret, frame=capture.read()
    height,width,_ = frame.shape

    for processor in processors:
        processor.process(frame)
    cv2.imshow("original", frame)
    for processor in processors:
        name, frame = processor.getResult()
        cv2.imshow(name, frame)
    key = cv2.waitKey(30)

    if key == 27:
        break

capture.release()
cv2.destroyAllwindows()
