import numpy as np
from imutils import face_utils
import dlib
import cv2


def landmark_detect(image):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('./face_detect/shape_predictor_68_face_landmarks.dat')
	# load the input image, resize it, and convert it to grayscale
	image = np.asarray(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	# loop over the face detections
	lmk_list = []
	if len(rects) > 0:
		rect = rects[0]
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		# (x, y, w, h) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the face number
		# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			lmk_list.append((x, y))
			# cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		return lmk_list
	else:
		return []
	# show the output image with the face detections + facial landmarks
	# cv2.imshow("Output", image)
	# cv2.waitKey(0)

from shapely.geometry import Point, Polygon

def get_convex_hull_coordinates(points):
	convex_hull = Polygon(points).convex_hull  # 构建凸包对象
	min_x, min_y, max_x, max_y = convex_hull.bounds  # 获取凸包的边界框

	hull_points = []
	for x in range(int(min_x), int(max_x) + 1):
		for y in range(int(min_y), int(max_y) + 1):
			point = Point(x, y)
			if point.within(convex_hull):
				hull_points.append(((x, y), 1.0))

	return hull_points
