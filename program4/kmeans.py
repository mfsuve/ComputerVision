import cv2
from matplotlib.pyplot import imshow, show, figure, title
import numpy as np
from random import randint


def kmeans(img, filter, numclusters):
	h = np.size(img, axis=0)
	w = np.size(img, axis=1)
	finish = False
	clustercenters = []
	prevcenters = []
	clusterpoints = {}

	# Initialization
	for i in range(numclusters):
		clustercenters.append(randint(0, 255))
		clusterpoints[i] = []
		prevcenters.append(-256)		# ensuring the error is big at first

	# Iteration
	while not finish:
		for x in range(h):
			for y in range(w):
				if filter[x, y] > 0:
					differences = [abs(img[x, y] - center) for center in clustercenters]
					centerindex = differences.index(min(differences))
					clusterpoints[centerindex].append((x, y))

		for i in range(numclusters):
			if len(clusterpoints[i]) > 0:
				clustercenters[i] = np.mean([img[p[0], p[1]] for p in clusterpoints[i]])
			else:
				clustercenters[i] = 0

		for points in clusterpoints.values():
			points[:] = []

		finish = True
		for c in range(len(clustercenters)):
			if abs(clustercenters[c] - prevcenters[c]) > 0.01:
				finish = False
			prevcenters[c] = clustercenters[c]

	newimage = np.zeros((h, w))
	for x in range(h):
		for y in range(w):
			if filter[x, y] > 0:
				differences = [abs(img[x, y] - center) for center in clustercenters]
				centerindex = differences.index(min(differences))
				newimage[x, y] = clustercenters[centerindex]

	return newimage


mr = cv2.imread('images/mr.jpg')
gray = cv2.cvtColor(mr, cv2.COLOR_BGR2GRAY)
figure(1)
title('Gray image')
imshow(gray, cmap='gray')
show()

threshold = gray > 75
figure(2)
title('Thresholded image')
imshow(threshold, cmap='gray')
show()

kernel = np.ones((25, 25), np.uint8)
threshold = np.array(threshold, np.uint8)
brain = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
figure(3)
title('Skull removed image')
imshow(brain, cmap='gray')
show()

# k-means with 2 cluster (brain and tumor)
segmented = kmeans(gray, brain, 2)
figure(5)
title('K-Means Segmented')
imshow(segmented, cmap='gray')
show()

tumor = np.array(segmented > 170, np.uint8)
tumor = cv2.Canny(tumor, 0, 1)
figure(6)
title('Tumor Edges')
imshow(tumor, cmap='gray')
show()

# I dialated the edges because they look too thin to detect sometimes
kernel = np.ones((2, 2), np.uint8)
tumor = cv2.dilate(tumor, kernel)

tumoroutlined = mr[:]
tumoroutlined[tumor > 0] = [0, 0, 255]
figure(7)
title('Tumor Outlined MR Image')
imshow(tumoroutlined, cmap='gray')
show()
