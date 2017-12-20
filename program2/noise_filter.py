import cv2
import numpy as np
from matplotlib import pyplot as plt

def meanFilter(img):
    # filterSize = 3
    h = np.size(img, axis=0)
    w = np.size(img, axis=1)
    newimage = np.zeros((h, w))
    # Extend the size for filter to fit
    img = np.append(img, [img[h-2, :]], axis=0)
    img = np.append([img[1, :]], img, axis=0)
    img = np.append(img, img[:, w-2].reshape(h+2, 1), axis=1)
    img = np.append(img[:, 1].reshape(h+2, 1), img, axis=1)
    # Filtering
    for x in range(1, w+1):
        for y in range(1, h+1):
            newimage[x - 1, y - 1] = np.sum(img[x - 1:x + 2, y - 1:y + 2]) / 9

    return newimage

def medianFilter(img):
    # filterSize = 3
    h = np.size(img, axis=0)
    w = np.size(img, axis=1)
    newimage = np.zeros((h, w))
    # Extend the size for filter to fit
    img = np.append(img, [img[h-2, :]], axis=0)
    img = np.append([img[1, :]], img, axis=0)
    img = np.append(img, img[:, w-2].reshape(h+2, 1), axis=1)
    img = np.append(img[:, 1].reshape(h+2, 1), img, axis=1)
    # Filtering
    for x in range(1, w+1):
        for y in range(1, h+1):
            newimage[x - 1, y - 1] = np.sort(img[x - 1:x + 2, y - 1:y + 2].reshape(9))[4]

    return newimage

def blend(mean, median, alpha):
    if alpha > 1 or alpha < 0:
        raise Exception("Alpha needs to be in range of [0, 1].")
    return mean * alpha + median * (1 - alpha)

gauss = cv2.imread('images/cameramanN1.jpg', 0)
impulse = cv2.imread('images/cameramanN2.jpg', 0)
both = cv2.imread('images/cameramanN3.jpg', 0)

# Arithmetic Mean Filter
meanGauss = meanFilter(gauss)
meanImpulse = meanFilter(impulse)
meanBoth = meanFilter(both)

plt.figure(1)
plt.imshow(meanGauss, cmap='gray')
plt.title('Mean Filter to Gaussian Noise')
plt.show()

plt.figure(2)
plt.imshow(meanImpulse, cmap='gray')
plt.title('Mean Filter to Impulsive Noise')
plt.show()

plt.figure(3)
plt.imshow(meanBoth, cmap='gray')
plt.title('Mean Filter to Both Noise')
plt.show()

# Median Filter
medianGauss = medianFilter(gauss)
medianImpulse = medianFilter(impulse)
medianBoth = medianFilter(both)

plt.figure(4)
plt.imshow(medianGauss, cmap='gray')
plt.title('Median Filter to Gaussian Noise')
plt.show()

plt.figure(5)
plt.imshow(medianImpulse, cmap='gray')
plt.title('Median Filter to Impulsive Noise')
plt.show()

plt.figure(6)
plt.imshow(medianBoth, cmap='gray')
plt.title('Median Filter to Both Noise')
plt.show()

# Blending
blendGauss = blend(meanGauss, medianGauss, 1)
blendImpulse = blend(meanImpulse, medianImpulse, 0)
blendBoth = blend(meanBoth, medianBoth, 0.5)

plt.figure(7)
plt.imshow(blendGauss, cmap='gray')
plt.title('Blending on Gaussian with alpha 1')
plt.show()

plt.figure(8)
plt.imshow(blendImpulse, cmap='gray')
plt.title('Blending on Impulsive with alpha 0')
plt.show()

plt.figure(9)
plt.imshow(blendBoth, cmap='gray')
plt.title('Blending on Both with alpha 0.5')
plt.show()
