# Alejandro and Premila work for 6.835
# Started on April 17th 2021
# skeleton code inspiration from: https://www.youtube.com/watch?v=brwgBf6VB0I
# list of landmarks in order: https://google.github.io/mediapipe/solutions/pose.html

import cv2
import time
import poseModule as pm
import csv
import math
from multiprocessing import Process
import mediapipe as mp
def parse_list_string(string):
	# get strings like this
	string = string.replace("]", "")
	string = string.replace("[", "")
	ret = string.split(",")
	ret = [string for string in ret]
	return ret


def get_data(trainer_file):
	squats_rows = []
	squats_times = []
	with open(trainer_file) as csv_file:
	    csv_reader = csv.reader(csv_file)
	    squats_rows = list(csv_reader)
	    squats_rows.pop(0) # get rid of the header of the csv
	    # fix squats_times
	    squats_times = [int(float(row[0])) for row in squats_rows]
	    # fix squats_rows
	    new = []
	    for row in squats_rows:
	    	new_row = []
	    	for element in row:
	    		new_row.append(parse_list_string(element))	# parse str of list back to list
	    	new_row.pop(0)									# get rid of time
	    	new.append(new_row)							
	    squats_rows = new 									# update squats_rows
	return 'squats',squats_times, squats_rows


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def calculate_accuracy(cv2, lmList, timestamp):
	accuracy = 0
	check_limbs = False
	rounding_factor = 2
	
	if len(lmList) > 28:
		right_hip = [int(lmList[24][1]), int(lmList[24][2])]
		right_knee = [int(lmList[26][1]), int(lmList[26][2])]
		right_ankle = [int(lmList[28][1]), int(lmList[28][2])] 
		user_angle = round(angle3pt(right_hip, right_knee, right_ankle ), rounding_factor)
		cv2.circle(frame_cam, (right_hip[0], right_hip[1]), 15, (0, 0, 255), cv2.FILLED)
		cv2.circle(frame_cam, (right_knee[0], right_knee[1]), 15, (0, 0, 255), cv2.FILLED)
		cv2.circle(frame_cam, (right_ankle[0], right_ankle[1]), 15, (0, 0, 255), cv2.FILLED)
		
		closest_index = squats_times.index(min(squats_times, key=lambda x:abs(x-timestamp))) # get index of closest time
		right_hip_trainer = [int(squats_rows[closest_index][24][1]), int(squats_rows[closest_index][24][2])]
		right_knee_trainer = [int(squats_rows[closest_index][26][1]), int(squats_rows[closest_index][26][2])]
		right_ankle_trainer = [int(squats_rows[closest_index][28][1]), int(squats_rows[closest_index][28][2])]
		trainer_angle = round(angle3pt(right_hip_trainer, right_knee_trainer, right_ankle_trainer), rounding_factor)

		error = abs(user_angle - trainer_angle)
		if error > 135:
			accuracy = 0
		else:
			accuracy = round((135 - error)/135, rounding_factor)
		print(user_angle, trainer_angle, accuracy)
		
	return accuracy

def project_trainer_skeleton(img, csv_data, timestamp):
	closest_index = squats_times.index(min(squats_times, key=lambda x:abs(x-timestamp))) # get index of closest time
	for body_part in csv_data[closest_index]:
		cx = int(body_part[1])
		cy = int(body_part[2])
		cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

	####### THIS DRAWS CONNECTING LINES BUT TAKES TOO LONG ######
	# mpDraw = mp.solutions.drawing_utils
	# imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# mpPose = mp.solutions.pose
	# pose = mpPose.Pose(False, False, True, 0.5, 0.5)
	# results = pose.process(imgRGB)
	# print(results.pose_landmarks)
	# if results.pose_landmarks:
	# 	mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
	#############################################################

	return img

identifier, squats_times, squats_rows = get_data('squats.csv')
current_milli_time = lambda: int(round(time.time() * 1000))

# camera feed
cap_cam = cv2.VideoCapture(0) 	# this captures live video from your webcam

# video feed
# filename = 'videos/squats.MOV'
filename = 'videos/squats500k.mp4'
cap_vid = cv2.VideoCapture(filename)
# Get length of the video.
fps = cap_vid.get(cv2.CAP_PROP_FPS)     # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frame_count = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
video_length = frame_count/fps * 1000 	# in milliseconds

width 	= int(720/3)  
height	= int(1280/3)
start 	= current_milli_time()
alpha 	= 0.1

pTime = 0
frame_counter = 0
detector = pm.poseDetector()
while True:
	# read from the camera
	success, frame_cam = cap_cam.read()
	time_passed = current_milli_time() - start # Capture the frame at the current time point
	frame_cam = cv2.flip(frame_cam,1)

	# read from the video
	ret = cap_vid.set(cv2.CAP_PROP_POS_MSEC, time_passed) # set time frame
	ret, frame_vid = cap_vid.read()
	
	# If the last frame is reached, reset the video
	if time_passed >= video_length:
		print("Resetting video")
		_ = cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to the first frame. Returns bool.
		start = current_milli_time()
		continue
	
    # find the skeleton
	frame_cam = detector.findPose(frame_cam)
	lmList = detector.findPosition(frame_cam, draw=False)

	accuracy = calculate_accuracy(cv2, lmList, time_passed)
	
	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	# add on other video
	frame_vid = project_trainer_skeleton(frame_vid, squats_rows, time_passed)
	frame_vid = cv2.resize(frame_vid, (height, width), interpolation = cv2.INTER_AREA)
	added_image = cv2.addWeighted(frame_cam[100:100+width,800:800+height,:],alpha,frame_vid[0:width,0:height,:],1-alpha,0)
	
	# Change the region with the result
	frame_cam[60:60+width,800:800+height] = added_image
	# For displaying current value of alpha(weights)
	font = cv2.FONT_HERSHEY_SIMPLEX
	display_string = 'frames:{} alpha:{} accuracy:{}'.format(round(fps,1), alpha, accuracy)
	cv2.putText(frame_cam,display_string,(20,40), font, 1.5,(0,255,0),4,cv2.LINE_AA)
	cv2.imshow('a',frame_cam)

	cv2.waitKey(1)



