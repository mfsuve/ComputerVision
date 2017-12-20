import cv2
import numpy as np
from matplotlib.pyplot import imshow, show, title


def estimateA(A, b):
    At = np.transpose(A)

    AtA = np.dot(At, A)
    invAtA = np.linalg.inv(AtA)

    first = np.dot(invAtA, At)
    a = np.dot(first, b)

    return a


def affine(image, A):
    h = np.size(image, axis=1)
    w = np.size(image, axis=0)

    m = np.zeros((h * w, 2))
    for i in range(h):
        for j in range(w):
            m[j + w * i] = [i, j]

    # invA = np.linalg.inv(A)
    LUT = np.dot(m, A)

    newh = int(max(LUT[:, 1])) + 1
    neww = int(max(LUT[:, 0])) + 1

    newimg = np.zeros((newh, neww))
    for i in range(h):
        for j in range(w):
            x = int(LUT[j + w * i, 0])
            y = int(LUT[j + w * i, 1])
            newimg[y, x] = image[j, i]

    return newimg


cameraman1 = cv2.imread('images/cameraman1.jpg', 0)
cameraman2 = cv2.imread('images/cameraman2.jpg', 0)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(cameraman1, None)
kp2, des2 = sift.detectAndCompute(cameraman2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
correct = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        correct.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img = cv2.drawMatchesKnn(cameraman1, kp1, cameraman2, kp2, correct, None, flags=2)

title('Matching points')
imshow(img)
show()

pts1 = [kp1[p[0].queryIdx].pt for p in correct]
pts2 = [kp2[p[0].trainIdx].pt for p in correct]

p1 = []
p2 = []
for i in range(len(pts1)):
    x1 = pts1[i][1]
    x2 = pts2[i][1]
    if abs(x1 - x2) < 1:
        p1.append(pts1[i])
        p2.append(pts2[i])

M = estimateA(p1, p2)

newimage = affine(cameraman1, M)
title('Transformed image')
imshow(newimage, cmap='gray')
show()
