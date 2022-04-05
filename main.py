import cv2 as cv
from matplotlib import pyplot as plt

# image provided by university course material
img_original = cv.imread('Kakadu_National_Park.JPG')
img1 = cv.resize(img_original, (480, 640), 3)

# Check whether the image is grayscale or colour
if (img1.any()) != None:
    if len(img1.shape) == 2:
        print('grayscale, we do not need to convert the image')
        new_img = img1
    elif len(img1.shape) == 3:
        new_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        print('Coloured image provided, image converted to grayscale')
else:
    print("Cannot find image")

# Canny edge detector
img_canny = cv.Canny(new_img, 50, 150)

# Sobel Filter
img_sobelx = cv.Sobel(new_img, cv.CV_64F, 1, 0, ksize=3)
img_sobely = cv.Sobel(new_img, cv.CV_64F, 0, 1, ksize=3)
abs_sobelx = cv.convertScaleAbs(img_sobelx)
abs_sobely = cv.convertScaleAbs(img_sobely)
Sobel_filter = cv.bitwise_or(abs_sobelx, abs_sobely)

# Display all the images
plt.subplot(1, 3, 1), plt.imshow(new_img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(img_canny, cmap='gray')
plt.title('Canny Edge Detector'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(Sobel_filter, cmap='gray')
plt.title('Sobel filter'), plt.xticks([]), plt.yticks([])
plt.show()
