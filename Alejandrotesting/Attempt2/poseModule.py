# Alejandro and Premila work for 6.835
# Started on April 17th 2021
# skeleton code inspiration from: https://www.youtube.com/watch?v=brwgBf6VB0I
# list of landmarks in order: https://google.github.io/mediapipe/solutions/pose.html

import cv2
import mediapipe as mp
import time

class poseDetector():
	
	def __init__(self, mode = False, upperBody = False, smooth = True, detectionConfidence = 0.5, trackConfidence = 0.5):
		self.mode = mode
		self.upperBody = upperBody
		self.smooth = smooth
		self.detectionConfidence = detectionConfidence
		self.trackConfidence = trackConfidence

		self.mpDraw = mp.solutions.drawing_utils
		self.mpPose = mp.solutions.pose
		self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smooth, self.detectionConfidence, self.trackConfidence)


	def findPose(self, img, draw=True):

		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.pose.process(imgRGB)
		# print(results.pose_landmarks)
		if self.results.pose_landmarks:
			if draw:
				self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
		
		return img

	def findPosition(self, img, draw=True):
		# list of landmarks in order: https://google.github.io/mediapipe/solutions/pose.html
		landmark_list = []
		if self.results.pose_landmarks:
			for id, lm in enumerate(self.results.pose_landmarks.landmark):
				h, w, c = img.shape
				# print(id, lm)
				cx, cy = int(lm.x * w), int(lm.y * h)
				landmark_list.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

		return landmark_list
	

def main():
	# cap = cv2.VideoCapture('videos/sample_video.mp4') 
	cap = cv2.VideoCapture(0) 	# gets live video feed
	pTime = 0
	detector = poseDetector()
	while True:
		success, img = cap.read()
		img = detector.findPose(img)
		# list of landmarks in order: https://google.github.io/mediapipe/solutions/pose.html
		lmList = detector.findPosition(img)

		body_part = lmList[14] # 14 is right elbow
		print(body_part)
		cv2.circle(img, (body_part[1], body_part[2]), 15, (0, 0, 255), cv2.FILLED)

		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime

		cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
		cv2.imshow("Image", img)

		cv2.waitKey(1)

if __name__ == "__main__":
	main()