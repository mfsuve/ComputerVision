import cv2
import numpy as np
import math

Icolor = cv2.imread('images/mustafa.jpeg')

height = np.size(Icolor, 0)
width = np.size(Icolor, 1)

Igray = np.zeros((height, width))

Igray[:,:] = Icolor[:,:,0]
Igray[:,:] += Icolor[:,:,1]
Igray[:,:] += Icolor[:,:,2]
Igray[:,:] /= 3

cv2.imwrite("gray.png", Igray)

# Test

max = np.amax(Igray)
min = np.amin(Igray)

# max = 249.0 , min = 0.666666666667
print("max =", max, ", min =", min)

print("This image can be represented with at least", math.ceil(math.log(max-min, 2)), "bits.")