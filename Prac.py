import cv2
import numpy as np
class prac:
    def __init__(self):
        pass
    def process(self,frame):
        mask = frame
        #threshold mask
        ret, foreground = cv2.threshold(mask, 200, 300, cv2.THRESH_BINARY)
        ret, shadow = cv2.threshold(mask, 200, 300, cv2.THRESH_TOZERO)


        # stack images vertically
        self.result = np.concatenate((foreground,shadow),axis=1)


    def getResult(self):
        return("prac", self.result)