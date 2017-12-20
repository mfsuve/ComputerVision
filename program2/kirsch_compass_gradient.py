import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

def initFilters():
    N = np.array([-3, -3, -3, -3, 0, -3, 5, 5, 5]).reshape(3, 3)
    W = np.array([-3, -3, 5, -3, 0, 5, -3, -3, 5]).reshape(3, 3)
    S = np.array([5, 5, 5, -3, 0, -3, -3, -3, -3]).reshape(3, 3)
    E = np.array([5, -3, -3, 5, 0, -3, 5, -3, -3]).reshape(3, 3)

    NW = np.array([-3, -3, -3, -3, 0, 5, -3, 5, 5]).reshape(3, 3)
    SW = np.array([-3, 5, 5, -3, 0, 5, -3, -3, -3]).reshape(3, 3)
    SE = np.array([5, 5, -3, 5, 0, -3, -3, -3, -3]).reshape(3, 3)
    NE = np.array([-3, -3, -3, 5, 0, -3, 5, 5, -3]).reshape(3, 3)

    return {'N': N, 'W': W, 'S': S, 'E': E, 'NW': NW, 'SW': SW, 'SE': SE, 'NE': NE}

def applyFilter(img, Filter):
    # filterSize = 3
    reverseFilter = Filter[::-1, ::-1]
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
            val = np.sum(np.multiply(img[x - 1:x + 2, y - 1:y + 2], reverseFilter))
            newimage[x - 1, y - 1] = max(0, val)

    return newimage

def maxGradient(img, filters):
    h = np.size(img, axis=0)
    w = np.size(img, axis=1)

    images = np.zeros((len(filters.values()), h, w))
    i = 0
    for f in filters.values():
        images[i] = applyFilter(img, f)
        i += 1

    maxResponse = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            val = np.max(images[:, x, y])
            maxResponse[x, y] = val > 400

    return maxResponse

def direction(img, filters):
    h = np.size(img, axis=0)
    w = np.size(img, axis=1)

    global stepSize

    images = np.zeros((len(filters.values()) + 1, h, w))
    images[1] = applyFilter(img, filters.get('N'))
    images[2] = applyFilter(img, filters.get('E'))
    images[3] = applyFilter(img, filters.get('S'))
    images[4] = applyFilter(img, filters.get('W'))
    images[5] = applyFilter(img, filters.get('NE'))
    images[6] = applyFilter(img, filters.get('SE'))
    images[7] = applyFilter(img, filters.get('NW'))
    images[8] = applyFilter(img, filters.get('SW'))

    U = np.zeros((h, w))
    V = np.zeros((h, w))
    dir = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            values = images[:, x, y]
            val = np.max(values)
            index = np.argmax(values)
            dir[x, y] = index if val > 400 else 0
            if dir[x, y] == 1:
                V[x, y] = 1
            elif dir[x, y] == 2:
                U[x, y] = 1
            elif dir[x, y] == 3:
                V[x, y] = -1
            elif dir[x, y] == 4:
                U[x, y] = -1
            elif dir[x, y] == 5:
                U[x, y] = sqrt(2) / 2
                V[x, y] = sqrt(2) / 2
            elif dir[x, y] == 6:
                U[x, y] = sqrt(2) / 2
                V[x, y] = -sqrt(2) / 2
            elif dir[x, y] == 7:
                U[x, y] = -sqrt(2) / 2
                V[x, y] = sqrt(2) / 2
            elif dir[x, y] == 8:
                U[x, y] = -sqrt(2) / 2
                V[x, y] = -sqrt(2) / 2


    return dir, U[stepSize::stepSize, stepSize::stepSize], V[stepSize::stepSize, stepSize::stepSize]

def gradient(img):
    h = np.size(img, axis=0)
    w = np.size(img, axis=1)

    global stepSize

    gradients = np.zeros((h, w))
    U = np.zeros((h, w))
    V = np.zeros((h, w))
    for x in range(1, h-1):
        for y in range(1, w-1):
            u = (int(img[x+1, y]) - int(img[x-1, y])) / 2
            v = (int(img[x, y+1]) - int(img[x, y-1])) / 2
            val = sqrt(u**2 + v**2)
            if val > 15:
                gradients[x, y] = val
                U[x, y] = u
                V[x, y] = v

    ### This part was used without the treshold
    ### To see the histogram of the gradient values
    # m = np.max(gradients)
    # a = np.arange(m + 1)
    # c = gradients.reshape(h*w)
    # b, x = np.histogram(c, np.size(a))
    # plt.plot(a, b)
    # plt.show()

    return gradients, U[stepSize::stepSize, stepSize::stepSize], V[stepSize::stepSize, stepSize::stepSize]

# Program
I = cv2.imread('images/StairsBuildingsN.png', 0)
filters = initFilters()

for f in filters.keys():
    plt.imshow(applyFilter(I, filters.get(f)), cmap='gray')
    plt.title('%s filter' % f)
    plt.show()

Jmag = maxGradient(I, filters)
plt.imshow(Jmag, cmap='gray')
plt.title('Max Gradient')
plt.show()

global stepSize
stepSize = 4

Jdir, U, V = direction(I, filters)
X, Y = np.meshgrid(np.arange(stepSize, 512, stepSize), np.arange(stepSize, 512, stepSize))

plt.imshow(Jmag, cmap='gray')
plt.quiver(X, Y, U, V, scale=30, color='r', units='width')
plt.title('Vector Direction')
plt.show()

Imag, U, V = gradient(I)

plt.imshow(Imag, cmap='gray')
plt.quiver(X, Y, V, U, scale=1750, color='r', units='width')
plt.title('Gradient Vector Direction')
plt.show()

I = applyFilter(I, np.ones((3, 3)) / 9)

plt.imshow(I, cmap='gray')
plt.title('I with Gaussian Filter')
plt.show()

for f in filters.keys():
    plt.imshow(applyFilter(I, filters.get(f)), cmap='gray')
    plt.title('%s filter' % f)
    plt.show()

Jmag = maxGradient(I, filters)
plt.imshow(Jmag, cmap='gray')
plt.title('Max Gradient')
plt.show()

Jdir, U, V = direction(I, filters)
X, Y = np.meshgrid(np.arange(stepSize, 512, stepSize), np.arange(stepSize, 512, stepSize))

plt.imshow(Jmag, cmap='gray')
plt.quiver(X, Y, U, V, scale=30, color='r', units='width')
plt.title('Vector Direction')
plt.show()

Imag, U, V = gradient(I)

plt.imshow(Imag, cmap='gray')
plt.quiver(X, Y, V, U, scale=1750, color='r', units='width')
plt.title('Gradient Vector Direction')
plt.show()
