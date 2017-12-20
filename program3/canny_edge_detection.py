from cv2 import imread, Canny
from math import exp, sqrt, atan2, pi
import numpy as np
from matplotlib.pyplot import imshow, show, title


def gauss_func(sig, x, y):
    return exp(-((x**2 + y**2)/(2 * (sig**2))))


def create_gaussian_filter(sig, size):
    f = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            f[x, y] = gauss_func(sig, x - (size - 1) / 2, y - (size - 1) / 2)

    # I return this to make the sum equal to 1
    return f / np.sum(f)


def gauss_filter(I, sig, size):
    # since f is symmetric from all angles, we do not need to reverse it
    f = create_gaussian_filter(sig, size)
    I = np.lib.pad(I, ((0, size), (0, size)), 'reflect')

    h = np.size(I, axis=0)
    w = np.size(I, axis=1)
    newimage = np.zeros((h, w))
    newimage[:size, :size] = f

    FI = np.fft.fft2(I)
    Fnewimage = np.fft.fft2(newimage)
    newimage = np.fft.ifft2(FI * Fnewimage)

    return abs(newimage[(size+1)//2:-(size-1)//2, (size+1)//2:-(size-1)//2])


def gradient(I, tau):
    h = np.size(I, axis=0)
    w = np.size(I, axis=1)
    M = np.zeros((h, w))
    A = np.zeros((h, w))

    for x in range(1, h - 1):
        for y in range(1, w - 1):
            Ix = (I[x + 1, y] - I[x - 1, y]) / 2
            Iy = (I[x, y + 1] - I[x, y - 1]) / 2
            dist = sqrt(Ix**2 + Iy**2)
            M[x, y] = dist if dist > tau else 0
            angle = atan2(Iy, Ix)
            angle += pi / 2
            # to keep the angle between [0, 2pi]
            if angle < 0:
                angle += 2 * pi
            if angle > 2*pi:
                angle -= 2 * pi
            A[x, y] = angle

    return M, A


def nonmaxima(M, A):
    h = np.size(M, axis=0)
    w = np.size(M, axis=1)
    m = np.zeros((h, w))
    s = pi/8
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            angle = A[x, y]
            if (angle < s or angle >= 15*s) or (angle < 9*s and angle >= 7*s): #lr
                if(M[x, y] >= M[x, y + 1] and M[x, y] >= M[x, y - 1]):
                    m[x, y] = M[x, y] > 0
            elif (angle < 5*s and angle >= 3*s) or (angle < 13*s and angle >= 11*s): #ud
                if(M[x, y] >= M[x + 1, y] and M[x, y] >= M[x - 1, y]):
                    m[x, y] = M[x, y] > 0
            elif (angle < 3*s and angle >= s) or (angle < 11*s and angle >= 9*s): #ur
                if(M[x, y] >= M[x - 1, y + 1] and M[x, y] >= M[x + 1, y - 1]):
                    m[x, y] = M[x, y] > 0
            elif (angle < 7*s and angle >= 5*s) or (angle < 15*s and angle >= 13*s): #ul
                if(M[x, y] >= M[x + 1, y + 1] and M[x, y] >= M[x - 1, y - 1]):
                    m[x, y] = M[x, y] > 0

    return m


def canny(I, sig, tau):
    I = gauss_filter(I, sig, 9)
    M, A = gradient(I, tau)
    E = nonmaxima(M, A)
    return E, M, A

wirebond = imread('images/Fig2wirebond_mask.jpg', 0)
E, M, A = canny(wirebond, sqrt(0.5), 5)

title('wirebond (sigma)^2 = 0.5')
imshow(E, cmap='gray')
show()

E, M, A = canny(wirebond, sqrt(1), 5)

title('wirebond (sigma)^2 = 1')
imshow(E, cmap='gray')
show()

E, M, A = canny(wirebond, sqrt(3), 5)

title('wirebond (sigma)^2 = 3')
imshow(E, cmap='gray')
show()

bottles = imread('images/Fig3bottles.jpg', 0)
E, M, A = canny(bottles, sqrt(50), 7.5)

title('Edge map Bottles (sigma)^2 = 50, tau = 7.5')
imshow(E, cmap='gray')
show()

img = imread('images/mustafa.jpeg')
grayimg = np.dot(img[:, :, :3], [0.299, 0.587, 0.114])

M, A = gradient(grayimg, 10)

title('Mustafa image, Gradient with threshold 10')
imshow(M, cmap='gray')
show()

# Built-in Canny
cannyimg = Canny(img, 100, 200)

title('Mustafa image, Edge with Built-in Canny')
imshow(cannyimg, cmap='gray')
show()
