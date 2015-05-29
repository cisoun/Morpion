#!/usr/bin/env python2.7

'''
Tic tac toe detector

Some methods are taken from there:
	http://stackoverflow.com/questions/14248571/finding-properties-of-sloppy-hand-drawn-rectangles
'''

import cv2
import cv2.cv as cv
import getopt
import numpy as np
import sys
import zhangsuen
from operator import itemgetter

ADAPTIVETHRESHOLD = False
DEVICE = 0
DEBUG = False
THRESHOLD = 60
SIZE_MIN = 1000
SIZE_MAX = 20000

grid = []

class Square:
	def __init__(self, contours):
		self.contours = contours
		self.moment = cv2.moments(contours)
		self.cx = int(self.moment['m10'] / self.moment['m00'])
		self.cy = int(self.moment['m01'] / self.moment['m00'])

	def __repr__(self):
		return '[' + str(self.cx) + ', ' + str(self.cy) + '] ' + str(self.contours.tolist())

def nothing(x):
	pass

def angle_cos(p0, p1, p2):
	d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
	return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

'''
Square detection
'''
def find_squares(img, size_min, size_max):
	# Squares detection process
	squares = []
	for gray in cv2.split(img):
		for thrs in xrange(0, 255, 26):
			if thrs == 0:
				bin = cv2.Canny(gray, 0, 100, apertureSize=5)
				bin = cv2.dilate(bin, None)
			else:
				retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
			contours, hierarchy = cv2.findContours(
			    bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			for cnt in contours:
				cnt_len = cv2.arcLength(cnt, True)
				cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
				if len(cnt) == 4 and cv2.contourArea(cnt) > size_min and cv2.contourArea(cnt) < size_max and cv2.isContourConvex(cnt):
					c = cnt.reshape(-1, 2)
					max_cos = np.max(
					    [angle_cos(c[i], c[(i + 1) % 4], c[(i + 2) % 4]) for i in xrange(4)])
					if max_cos < 1:
						squares.append(Square(cnt))
	return squares

'''
Sort the grid from the top-left to the bottom-right.
'''
def sort_squares(grid):
	squares = []

	# Create a sortable array
	for s in grid:
		cx = s.cx
		cy = s.cy
		squares.append([grid.index(s), cx, cy])

	# Sorte by columns and rows
	sorted_grid = sorted(squares, key=itemgetter(2))
	lines = []
	for i in range(0, 3):
		lines.append(sorted(sorted_grid[i * 3 : i * 3 + 3], key=itemgetter(1)))

	# Reorder
	new_grid = []
	for l in lines:
		for s in l:
			new_grid.append(grid[s[0]])

	return new_grid

'''
Find the symbol in a specified cell
'''
def get_symbol(square, img, index):
	# Cut the part of the image associated to the cell
	x,y,w,h = cv2.boundingRect(square.contours)
	cell = img[y + 10 : y + h - 10, x + 10 : x + w - 10] # Crop a bit in order to remove the borders

	# Circles detection
	circles = cv2.HoughCircles(cell, cv.CV_HOUGH_GRADIENT, 1, 10, param1=20, param2=15, minRadius=10,maxRadius=40)

	# Squares detection
	# Very basic method : if the cell is filled and is not circle, it's a square.
	width, height = cell.shape[:2]
	ratio = float(cv2.countNonZero(cell)) / (width * height)

	# Assign symbol
	symbol = ''
	symbol = 'O' if circles != None else symbol
	symbol = 'X' if ratio < 1 and circles == None else symbol

	# Debug purpose
	# Show each cells
	if DEBUG:
		if circles != None:
			circles = np.uint16(np.around(circles))
			for i in circles[0,:]:
			    # draw the outer circle
			    cv2.circle(cell,(i[0],i[1]),i[2],(0,255,0),2)
			    # draw the center of the circle
			    cv2.circle(cell,(i[0],i[1]),2,(0,0,255),3)
		cv2.imshow(str(index), cell)

	return symbol

'''
Pre-process an image in order to analyze it.
'''
def preprocess(img):
	# Equalize
	img = cv2.equalizeHist(img)

	# Smoothen the image
	img = cv2.GaussianBlur(img, (11, 11), 0)

	# Keep a specified threshold
	if ADAPTIVETHRESHOLD:
		img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3) # Keep lines
	else:
		ret, img = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)


	# Remove parasites (not necessary)
	#kernel = np.ones((4,4),np.uint8)
	#img = cv2.dilate(img,kernel,iterations = 1) # Removing parasites

	# Skeleton
	zhangsuen.thin(img, False, False, False)

	# Debug purpose
	if DEBUG:
		cv2.imshow('preprocess', img)

	return img

'''
Process the image.
Show detected cells and their content.
 - Red for crosses
 - Blue for circles
 - Green for empty cells
'''
def process(img):
	grid = []

	# Work on a 500xXXX picture
	width, height = img.shape[:2]
	fx = 500 / width
	img = cv2.resize(img, (0, 0), fx=fx, fy=fx)

	# Pre-process the picture
	imgProcessed = preprocess(img)

	# Grid detection
	grid = find_squares(imgProcessed, SIZE_MIN, SIZE_MAX)
	grid = sort_squares(grid)

	# Show the grid and the cells
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	for i in range(0, len(grid)):
		symbol = get_symbol(grid[i], imgProcessed, i)
		# Cells
		s = grid[i]
		cv2.drawContours(img, [s.contours], 0, ((255 if symbol == 'O' else 0), 100, 255 if symbol == 'X' else 0), 2)
		# Cells center
		cx = s.cx
		cy = s.cy
		cv2.circle(img, (cx, cy), 3, (0, 0, 255), 2)

	cv2.imshow('morpion', img)

if __name__ == '__main__':

	try:
		opts, args = getopt.getopt(sys.argv[1:],"di:t:",["at", "device=", "threshold="])
	except getopt.GetoptError:
		print('morpion.py -d -i <device> -t <threshold> --at')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('morpion.py -d -i <device> -t <threshold> --at')
			sys.exit()
		elif opt == '-d':
			DEBUG = True
		elif opt in ("-at", "--at"):
			ADAPTIVETHRESHOLD = True
		elif opt in ("-i", "--device"):
			DEVICE = int(arg)
		elif opt in ("-t", "--threshold"):
			THRESHOLD = int(arg)



	# Capture a video
	video_capture = cv2.VideoCapture(DEVICE)
	cv2.namedWindow('morpion')
	# Create trackbar
	cv2.createTrackbar('Threshold', 'morpion', 1, 300, nothing)
	cv2.setTrackbarPos('Threshold', 'morpion', int(THRESHOLD))

	while True:
		# Capture frame-by-frame
		ret, frame = video_capture.read()
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if DEBUG:
			cv2.imshow('original', img)
		process(img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		THRESHOLD = cv2.getTrackbarPos('Threshold', 'morpion')
	video_capture.release()

	# Use an image
	#img = cv2.imread('grid-filled.jpg', cv2.IMREAD_GRAYSCALE)
	#img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
	#process(img)

	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()
