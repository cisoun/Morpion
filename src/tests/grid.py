# Grid detection :
# 	http://www.aishack.in/tutorials/sudoku-grabber-with-opencv-detection/

import numpy as np
import cv2

"""
Hough class utility

Algorithm take from
https://github.com/Itseez/opencv/blob/master/samples/cpp/tutorial_code/ImgTrans/HoughLines_Demo.cpp#L33
"""
class Hough:

	@staticmethod
	def standard(canny, dst, threshold):
		lines = cv2.HoughLines(canny, 1, np.pi / 180, threshold, 0, 0)
		for rho, theta in lines[0]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1 = int(x0 + 1000 * (-b))
			y1 = int(y0 + 1000 * (a))
			x2 = int(x0 - 1000 * (-b))
			y2 = int(y0 - 1000 * (a))

			cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)

	@staticmethod
	def probabilistic(canny, dst, threshold):
		lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold, 1000, 1)
		for line in lines[0]:
			cv2.line(dst, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 3)


"""
Grid detection
"""

kernel = np.array([[0, 1, 0],
				   [1, 1, 1],
				   [0, 1, 0]], dtype = np.uint8)

# Open our test grid.
src = cv2.imread('grid-empty.jpg', cv2.IMREAD_GRAYSCALE)

# Image processing
srcBlur = cv2.GaussianBlur(src, (11, 11), 0)
srcThreshold = cv2.adaptiveThreshold(srcBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
srcInverse = cv2.bitwise_not(srcThreshold)
srcDilate = cv2.dilate(srcInverse, kernel)

# Apply Hough for line detection
Hough.standard(srcDilate, src, 200)
cv2.imshow('Hough', src)

# Show results
while(True):
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()