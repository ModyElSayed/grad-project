from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import RPi.GPIO as GPIO


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


# Pin Definitions
output_pin = 18  # BCM pin 18, BOARD pin 12

# Pin Setup:
GPIO.setmode(GPIO.BCM)  # BCM pin-numbering scheme from Raspberry Pi
# set pin as an output pin with optional initial state of HIGH
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
	"models/shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0

file = open("testfile.txt", "w")
file.write("Starting demo now! Press CTRL+C to exit")

try:
	while True:
		ret, frame = cap.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = detect(gray, 0)

		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			if ear < thresh:
				flag += 1
				file.write("Flag:", flag)

				if flag >= frame_check:
					file.write("Drowsy, current flag value {} to pin {}".format(flag, output_pin))
					file.write("Drowsy")

					curr_value = GPIO.HIGH
					GPIO.output(output_pin, curr_value)
			else:
				flag = 0
				curr_value = GPIO.LOW
				GPIO.output(output_pin, curr_value)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

finally:
	GPIO.cleanup()
	file.close()

	cv2.destroyAllWindows()
	cap.release()

