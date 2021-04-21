import cv2
import time
import numpy as np

# the web camera frame has this resolution: (720, 1280, 3)

current_milli_time = lambda: int(round(time.time() * 1000))

# Camera feed
cap_cam = cv2.VideoCapture(0)
ret, frame_cam = cap_cam.read()

# Video feed
filename = 'videos/squats.MOV'
cap_vid = cv2.VideoCapture(filename)
ret, frame_vid = cap_vid.read()

# Resize the camera frame to the size of the video
width =     int(720/3)  
height =    int(1280/3)

# Starting from now, syncronize the videos
start = current_milli_time()
alpha = 0.15

while True:
    # Capture the next frame from camera
    ret, frame_cam = cap_cam.read()
    
    # Capture the frame at the current time point
    time_passed = current_milli_time() - start
    
    ret = cap_vid.set(cv2.CAP_PROP_POS_MSEC, time_passed)
    ret, frame_vid = cap_vid.read()
    frame_cam = cv2.flip(frame_cam,1)
    # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
    print(frame_cam.shape)
    frame_vid = cv2.resize(frame_vid, (height, width), interpolation = cv2.INTER_AREA)
    added_image = cv2.addWeighted(frame_cam[100:100+width,800:800+height,:],alpha,frame_vid[0:width,0:height,:],1-alpha,0)
    # Change the region with the result
    frame_cam[60:60+width,800:800+height] = added_image
    # For displaying current value of alpha(weights)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_cam,'alpha:{}'.format(alpha),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('a',frame_cam)
    

    k = cv2.waitKey(10)
    # Press q to break
    if k == ord('q'):
        break
    # press a to increase alpha by 0.1
    if k == ord('a'):
        alpha +=0.1
        if alpha >=1.0:
            alpha = 1.0

cap_cam.release()
cap_vid.release()
cv2.destroyAllWindows()

