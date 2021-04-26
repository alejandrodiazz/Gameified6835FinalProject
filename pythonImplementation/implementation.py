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


def index_of_body_part(body_part):
	body_parts_list = ["nose", "left_eye_inner", "left_eye", "left_eye_outer",
		"right_eye_inner", "right_eye", "right_eye_outer","left_ear", "right_ear",
		"mouth_left","mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
		"right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", 
		"left_index", "right_index", "left_thumb", "right_thumb","left_hip", 
		"right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", 
		"left_heel", "right_heel", "left_foot_index", "right_foot_index"]
	body_parts_dict = dict()
	for i, part in enumerate(body_parts_list):
		body_parts_dict[part] = i
	# print(body_parts_dict)
	return body_parts_dict[body_part]


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
	return 'exercise',squats_times, squats_rows


def angle3pt(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
        Returns a float between 0.0 and 360.0"""
    ang = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def calculate_angle_accuracy(img, lmList, to_compare, trainer_rows, trainer_times, timestamp):
	accuracies = list()
	rounding_factor = 2

	for body_parts in to_compare:
		i1 = index_of_body_part(body_parts[0])
		i2 = index_of_body_part(body_parts[1])
		i3 = index_of_body_part(body_parts[2])
		bodypart1 = [int(lmList[i1][1]), int(lmList[i1][2])]
		bodypart2 = [int(lmList[i2][1]), int(lmList[i2][2])]
		bodypart3 = [int(lmList[i3][1]), int(lmList[i3][2])] 

		if bodypart1[1] > 1280 or bodypart2[1] > 1280 or bodypart3[1] > 1280:
			accuracies.append(0)
			print("OUT OF FRAME Y DIRECTION: ", body_parts)
			continue

		if bodypart1[0] > 720 or bodypart2[0] > 720 or bodypart3[0] > 720:
			accuracies.append(0)
			print("OUT OF FRAME X DIRECTION: ", body_parts)
			continue

		user_angle = round(angle3pt(bodypart1, bodypart2, bodypart3 ), rounding_factor)		# find angle of user
		img = cv2.circle(img, (bodypart1[0], bodypart1[1]), 15, (0, 0, 255), cv2.FILLED)	# draw points of interest
		img = cv2.circle(img, (bodypart2[0], bodypart2[1]), 15, (0, 0, 255), cv2.FILLED)
		img = cv2.circle(img, (bodypart3[0], bodypart3[1]), 15, (0, 0, 255), cv2.FILLED)
		
		closest_index = trainer_times.index(min(trainer_times, key=lambda x:abs(x-timestamp))) # get index of closest time
		bodypart1_trainer = [int(trainer_rows[closest_index][i1][1]), int(trainer_rows[closest_index][i1][2])]
		bodypart2_trainer = [int(trainer_rows[closest_index][i2][1]), int(trainer_rows[closest_index][i2][2])]
		bodypart3_trainer = [int(trainer_rows[closest_index][i3][1]), int(trainer_rows[closest_index][i3][2])]
		trainer_angle = round(angle3pt(bodypart1_trainer, bodypart2_trainer, bodypart3_trainer), rounding_factor)

		error = abs(user_angle - trainer_angle)
		if error > 135:
			accuracies.append(0)
		else:
			accuracies.append(round((135 - error)/135, rounding_factor))
		
	print(accuracies)
		
	return img, round(sum(accuracies)/len(accuracies), rounding_factor)

def project_trainer_skeleton(img, trainer_rows, trainer_times, timestamp):
	closest_index = trainer_times.index(min(trainer_times, key=lambda x:abs(x-timestamp))) # get index of closest time
	for body_part in trainer_rows[closest_index]:
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

def run(csv, video_file, to_compare, exercise):
	identifier, trainer_times, trainer_rows = get_data(csv)
	current_milli_time = lambda: int(round(time.time() * 1000))

	# camera feed
	cap_cam = cv2.VideoCapture(0) 	# this captures live video from your webcam

	# video feed
	cap_vid = cv2.VideoCapture(video_file)
	# Get length of the video.
	fps = cap_vid.get(cv2.CAP_PROP_FPS)     # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
	frame_count = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
	video_length = frame_count/fps * 1000 	# in milliseconds
	font = cv2.FONT_HERSHEY_SIMPLEX

	width 	= int(720/3)  
	height	= int(1280/3)
	start 	= current_milli_time()
	alpha 	= 0.1
	score = 0

	pTime = 0
	frame_counter = 0
	detector = pm.poseDetector()


	while current_milli_time() - start < 6000:
		success, frame_cam = cap_cam.read()
		frame_cam = cv2.flip(frame_cam,1)
		time_left = 6-int(round((current_milli_time() - start)/1000, 0))
		display_string = 'Ready? ' + exercise + ' starting in:{} seconds'.format(time_left)
		cv2.putText(frame_cam,display_string,(20,60), font, 2,(0,255,0),6,cv2.LINE_AA)
		cv2.imshow('a',frame_cam)
		cv2.waitKey(1)


	start 	= current_milli_time()
	while True:
		# read from the camera
		success, frame_cam = cap_cam.read()
		time_passed = current_milli_time() - start # Capture the frame at the current time point
		frame_cam = cv2.flip(frame_cam,1)

		# read from the video
		ret = cap_vid.set(cv2.CAP_PROP_POS_MSEC, time_passed) # set time frame
		ret, frame_vid = cap_vid.read()
		
	    # find the skeleton
		frame_cam = detector.findPose(frame_cam)
		lmList = detector.findPosition(frame_cam, draw=False)
		# print(lmList)
		if len(lmList) == 0:
			print("ERRRRORRRR: Length of list 0")
			continue

		frame_cam, accuracy = calculate_angle_accuracy(img = frame_cam, lmList = lmList, to_compare = to_compare, trainer_rows = trainer_rows, trainer_times = trainer_times, timestamp = time_passed)
		score += accuracy

		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime

		# ADD ON OTHER VIDEO
		# If the last frame is reached, reset the video
		if time_passed >= video_length:
			print("Video Done")
			_ = cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to the first frame. Returns bool.
			start = current_milli_time()
			display_string = 'Good Job! SCORE:{} '.format(score)
			cv2.putText(frame_cam,display_string,(20,60), font, 2,(0,255,0),6,cv2.LINE_AA)
			cv2.imshow('a',frame_cam)
			cv2.waitKey(1)
			time.sleep(7)
			return

		frame_vid = project_trainer_skeleton(img = frame_vid, trainer_rows = trainer_rows, trainer_times = trainer_times, timestamp = time_passed)
		try:
			frame_vid = cv2.resize(frame_vid, (height, width), interpolation = cv2.INTER_AREA)
			added_image = cv2.addWeighted(frame_cam[100:100+width,800:800+height,:],alpha,frame_vid[0:width,0:height,:],1-alpha,0)
		except cv2.error:
			print("ERROR: could not create vid")
			time.sleep(2)
			continue

		# Change the region with the result
		frame_cam[60:60+width,800:800+height] = added_image
		# For displaying current value of alpha(weights)
		display_string = 'Frames:{} Accuracy:{}'.format(round(fps,1),accuracy)
		cv2.putText(frame_cam,display_string,(20,60), font, 2,(0,255,0),5,cv2.LINE_AA)
		cv2.imshow('a',frame_cam)

		cv2.waitKey(1)


def main():
	to_compare_squats = [["right_hip", "right_knee", "right_ankle"], ["right_shoulder","right_hip", "right_knee"]]
	to_compare_pushups = [["right_shoulder","right_hip", "right_ankle"], ["right_shoulder","right_elbow", "right_wrist"]]
	
	while True:
		run(csv = 'squats.csv', video_file= 'videos/squats200k.mp4', to_compare = to_compare_squats, exercise = "Squats")
		
		# run(csv = 'pushups.csv', video_file= 'videos/pushups200k.mp4', to_compare = to_compare_pushups, exercise = "Pushups")

if __name__ == "__main__":
	main()



