# Alejandro and Premila work for 6.835
# Started on April 17th 2021

import cv2
import time
import poseModule as pm

current_milli_time = lambda: int(round(time.time() * 1000))
# camera feed
cap_cam = cv2.VideoCapture(0) 	# this captures live video from your webcam
# video feed
filename = 'videos/squats.MOV'
cap_vid = cv2.VideoCapture(filename)
# Get length of the video.
fps = cap_vid.get(cv2.CAP_PROP_FPS)     # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frame_count = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
video_length = frame_count/fps * 1000 	# in milliseconds

width =     int(720/3)  
height =    int(1280/3)
start = current_milli_time()
alpha = 0.1

pTime = 0
frame_counter = 0
detector = pm.poseDetector()
while True:
	# read from the camera
	success, frame_cam = cap_cam.read()
	time_passed = current_milli_time() - start # Capture the frame at the current time point
	
	# read from the video
	ret = cap_vid.set(cv2.CAP_PROP_POS_MSEC, time_passed)
	ret, frame_vid = cap_vid.read()
	frame_cam = cv2.flip(frame_cam,1)
	
	# If the last frame is reached, reset the video
	if time_passed >= video_length:
		# Reset to the first frame. Returns bool.
		print("resetting video")
		_ = cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
		start = current_milli_time()
		# continue
	

    # find the skeleton
	frame_cam = detector.findPose(frame_cam)
	lmList = detector.findPosition(frame_cam, draw=False)
	if len(lmList) !=0:
		body_part = lmList[14] # 14 is right elbow
		print(body_part)
		cv2.circle(frame_cam, (body_part[1], body_part[2]), 15, (0, 0, 255), cv2.FILLED)

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	# add on other video
	frame_vid = cv2.resize(frame_vid, (height, width), interpolation = cv2.INTER_AREA)
	added_image = cv2.addWeighted(frame_cam[100:100+width,800:800+height,:],alpha,frame_vid[0:width,0:height,:],1-alpha,0)
	# Change the region with the result
	frame_cam[60:60+width,800:800+height] = added_image
	# For displaying current value of alpha(weights)
	font = cv2.FONT_HERSHEY_SIMPLEX
	display_string = 'frames:{} alpha:{}'.format(int(fps),alpha)
	cv2.putText(frame_cam,display_string,(20,40), font, 1,(255,0,0),2,cv2.LINE_AA)
	cv2.imshow('a',frame_cam)

	cv2.waitKey(1)


