import cv2
import numpy as np


img = cv2.imread("Frame/hello.JPG",1)
img = cv2.rectangle(img,(0,0),(510,128),(0,0,255),5)

# to put a circle in the picture
#img = cv2.circle(img,(447,63)(510,))

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()