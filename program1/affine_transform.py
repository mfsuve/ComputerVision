import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

normal = cv2.imread('images/cameraman.jpg')
# x=253 y=125 | x=401 y=221 | x=365 y=413 | 510 511 | 2   511
# x=212 y=135 | x=343 y=236 | x=378 y=408 | 506 509 | 164 464
affine = cv2.imread('images/cameramanAffine.jpg')

height = np.size(affine, axis=1)
width = np.size(affine, axis=0)

# matrix constructed by choosen coordinates of the normal image
A = np.array([[253, 125], [401, 221], [365, 413], [510, 511], [2, 511]])
# martix constructed by choosen x's and y's of the affine image
b = np.array([[212, 135], [343, 236], [378, 408], [506, 509], [164, 464]])
# transpose of A
At = np.transpose(A)

AtA = np.dot(At, A)
invAtA = np.linalg.inv(AtA)

first = np.dot(invAtA, At)
a = np.dot(first, b)

print(a)

A = np.zeros((width * height, 2))
for i in range(width):
    for j in range(height):
        A[i * height + j] = [i, j]

# A is now all the possible coordinates in order

inva = np.linalg.inv(a)
map = np.dot(A, inva)

# applying the transformation to the image
newimage = np.zeros((height, width, 3))
for i in range(width):
    for j in range(height):
        index = i * height + j
        x = math.floor(map[index, 0])
        y = math.floor(map[index, 1])
        if x >= width or x < 0 or y >= height or y < 0:
            continue
        newimage[j, i] = [256, 256, 256] - normal[y, x]

plt.figure(1)
plt.imshow(affine)
plt.show()












