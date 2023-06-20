from scipy.spatial import distance
from imutils import face_utils	#The module includes functions to work with facial landmarks
import imutils
import dlib
import cv2
import RPi.GPIO as GPIO
import keyboard
import time
import os


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


# Pin Definitions:
buzzer = 12
#led_pin = 13

GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
GPIO.setup(buzzer, GPIO.OUT, initial=GPIO.LOW)  # Buzzer pins set as output
#GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW)  # LED pins set as output

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()	#initialize detect This object is a pre-trained face detector that can detect human faces in an image using a Histogram of Oriented Gradients (HOG) feature-based method.
predict = dlib.shape_predictor(
	"models/shape_predictor_68_face_landmarks.dat")  # This object is a pre-trained facial landmark predictor that can detect 68 facial landmarks on a detected face. The predictor is loaded from a pre-trained model file named shape_predictor_68_face_landmarks.dat.

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]	# represent the indices of the left eye landmark, within the 68 facial landmarks detected by the predictor. 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)	#video capture
flag = 0

file = open("testfile.txt", "w")
file.write("Starting demo now! Press CTRL+C to exit\n")
file.write(f"Process ID: {os.getpid()}")


#def flash_led():
#	GPIO.output(led_pin, GPIO.HIGH)
#	time.sleep(0.5)
#	GPIO.output(led_pin, GPIO.LOW)


def change_frame():
	# A limit of # of frames can be implemented here
	global frame_check, file
	file.write("Button pressed\n" + "Current frame_check: " + str(frame_check) + "\n")
	while True:
		frame_check = input("Enter frame number: ")
		if frame_check.isdigit():
			frame_check = int(frame_check)
			break
		else:
			file.write("Please enter a number")
			#flash_led()

	file.write("New frame_check: " + str(frame_check) + "\n")

	print(frame_check)


keyboard.add_hotkey("backspace", change_frame)

try:
	while True:	# captures frames from the webcam, resizes them, converts them to grayscale, and detects faces in the frames.
		ret, frame = cap.read()	#Reads a frame from the video capture object cap returns two values: ret (a boolean indicating whether the frame was successfully read) and frame (the captured frame as a NumPy array).
		frame = imutils.resize(frame, width=450)	# Resizes the captured frame to a new width of 450 pixels while maintaining the aspect ratio.
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	#Converts the resized frame from the BGR color space to grayscale because the face detection and facial landmark detection algorithms work with grayscale images.
		subjects = detect(gray, 0)	#Applies the dlib frontal face detector (detect) to the grayscale image (gray). The second argument 0 indicates that the image should not be upsampled before detection. The result is a list of rectangles (subjects) representing the detected faces in the image.

		for subject in subjects:
			shape = predict(gray, subject)	#For each detected face (subject), the facial landmarks are predicted using the predict object
			shape = face_utils.shape_to_np(shape)  # the resulting shape is converted to NumPy Array
			leftEye = shape[lStart:lEnd]	#The left eye landmarks are extracted using the previously defined index ranges (lStart, lEnd).
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0	#The average EAR is computed by taking the mean of the left and right EARs.
			leftEyeHull = cv2.convexHull(leftEye)	#A convex hull is a geometric concept that, given a set of points, represents the smallest convex polygon that contains all of the points.
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)	#Draw the computed convexHull on the video
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			if ear < thresh:
				flag += 1
				file.write(f"Flag: {flag}\n")

				if flag >= frame_check:
					file.write("Drowsy, current flag value {} to pin {}\n".format(flag, buzzer))
					file.write("Drowsy\n")

					curr_value = GPIO.HIGH
					GPIO.output(buzzer, curr_value)
			else:
				flag = 0
				curr_value = GPIO.LOW
				GPIO.output(buzzer, curr_value)
			file.flush()

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

finally:
	GPIO.cleanup()
	file.close()
	cv2.destroyAllWindows()
	cap.release()
