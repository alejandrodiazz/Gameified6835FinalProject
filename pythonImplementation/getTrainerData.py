# Alejandro and Premila work for 6.835
# Started on April 17th 2021

import cv2
import time
import poseModule as pm
import csv					# to save data

csv_name = csv_name
cap = cv2.VideoCapture('videos/pushups.MOV')
# cap = cv2.VideoCapture(0) 	# this captures live video from your webcam
pTime = 0
detector = pm.poseDetector()
all_data = []
while True:
	try:
		success, img = cap.read()
		img = detector.findPose(img)
		lmList = detector.findPosition(img, draw=False)
		all_data.append(lmList)
		if len(lmList) !=0:
			body_part = lmList[14] # 14 is right elbow
			print(body_part)
			cv2.circle(img, (body_part[1], body_part[2]), 15, (0, 0, 255), cv2.FILLED)

		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime

		cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
		cv2.imshow("Image", img)

		cv2.waitKey(1)

	except: # reaches end of mp4 or mov file and saves trainer data

		with open(csv_name, mode='w+') as trainer_file:
		    trainer_writer = csv.writer(trainer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		    frame_rate = 30
		    seconds = 0
		    trainer_writer.writerow(["time"]+[str(number) for number in range(33)])
		    for index, line in enumerate(all_data):
		    	seconds = (1/30)*index*1000
		    	trainer_writer.writerow([seconds]+line)

		break
		    	







