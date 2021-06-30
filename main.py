import cv2
from tracker import *
import numpy as np
#import matplotlib.pyplot as plt
#from tracker import make_blobs

# reference of testvideo.mp4& testvideo2 below
# https://github.com/mailrocketsystems/AIComputerVision/blob/master/test_video.mp4
# https://mixkit.co/free-stock-video/quiet-tokyo-street-at-night-4451/


#create tracker object
dis_tracker = EuclideanDistTracker()

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 15
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 250
params.maxArea = 1500

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.05
# line 15 -41 seating up the blob dectecor
# Create a detector with the parameters
# OLD: detector = cv2.SimpleBlobDetector(params)
blobdetector = cv2.SimpleBlobDetector_create(params)

# capture object to read the frame from the video
capture = cv2.VideoCapture("Frame/testvideo2.mp4")



# is going to extract the moving object from the camera.
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)


# start loop to extract the frames one after another. on each loop we take one frame object detection from stable camera
while True:

    ret, frame = capture.read()

    height,width,_ = frame.shape


    #print(height,width)
    #regionOfInterest = frame[0:1800, 15:1280]
    # extract region of interest; height 340: 600 and width  500:700
    regionOfInterest = frame[0:1800, 100:1280]
    # Detect blobs.


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEY_POINTS ensures
# the size of the circle corresponds to the size of blob

# 1 Build Object Detection
# we need to apply the detection on the frame line 18
# mask shows every moving object detection such as car, moto and etc..
    mask = object_detector.apply(regionOfInterest)

    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    key_points = blobdetector.detect(frame)

    im_with_key_points = cv2.drawKeypoints(frame, key_points, np.array([]), (0, 0, 255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    detection = []
    for cnt in contours:

        # calculate the area amd remove small elements
        area = cv2.contourArea(cnt)
        # areaMin = cv2.getTrackbarPos("Area","Parameters")

        # if its greater than 100 pixel, we draw the contour, otherwise not..
        if area > 1500:
            cv2.drawContours(frame, cnt, -1, (255, 0, 255),2)
            # the contour is closed
            perimeter = cv2.arcLength(cnt, True)
            approximation = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            # print(len(approximation))
            # draw on the frame
            # draw all of the element -1; (255 green) & thickness 2
            # cv2. drawContours(regionOfInterest, [cnt],-1,(0,255,0),2)
            x, y, w, h = cv2.boundingRect(cnt)



            detection.append([x, y, w, h])

        # 2 Build Object Tracking
    boxes_ids = dis_tracker.update(detection)
    # print("box",boxes_ids)
    for box_id in boxes_ids:

        x, y, w, h, id = box_id
        # display the id's, -15 place object id higher the object; THE FONT,;
        # 2 increase the id size ;255 BLUE AND THICKNESS
        cv2.putText(regionOfInterest, str(id),(x+w+25,y+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,300),2)
        cv2.putText(regionOfInterest,"Area:" + str(int(area)), (x + w + 20,y+50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
        cv2.rectangle(regionOfInterest, (x,y), (x + w, y + h),(0,555,0),5)
    # try to cluster rectangle and draw them
    # show the frame on real time.
    cv2.imshow("regionOfInterest", regionOfInterest)
    cv2.imshow("Frame", frame)

    cv2.imshow("Mask", mask)

    cv2.imshow("blop",im_with_key_points)

    key = cv2.waitKey(30)

    if key == 27:
        break

capture.release()
cv2.destroyAllwindows()
