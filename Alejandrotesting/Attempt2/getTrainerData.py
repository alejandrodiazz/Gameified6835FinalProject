# Alejandro and Premila work for 6.835
# Started on April 17th 2021

import cv2
import time
import poseModule as pm

# cap = cv2.VideoCapture('sample_video.mp4')
cap = cv2.VideoCapture(0) 	# this captures live video from your webcam
pTime = 0
detector = pm.poseDetector()
while True:
	success, img = cap.read()
	img = detector.findPose(img)
	lmList = detector.findPosition(img, draw=False)
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