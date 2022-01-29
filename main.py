#import of openCV library
import cv2

#-----For jpeg image------#

#reading the image
image = cv2.imread('./resources/treesImage.jpeg')

#convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# save the "gray" image to ./resources/jpeg
cv2.imwrite('./resources/jpeg/treesImageGray.png', gray)

#The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced
#we apply this function to reduce the noise
"""
GaussianBlur(src, ksize, sigmaX)
src: input image
ksize: Gaussian Kernel Size. [height width]. 
       height and width should be odd and can have different values. 
       If ksize is set to [0 0], then ksize is computed from sigma values.
sigmaX: Kernel standard deviation along X-axis (horizontal direction).
"""
blur = cv2.GaussianBlur(gray, (13, 13), 0)
# save the "blur" image to ./resources/jpeg
cv2.imwrite('./resources/jpeg/treesImageBlur.png', blur)

"""
thresholding the image with the threshold(blur, 55, 255, cv2.CHAIN_APPROX_NONE) function
    blur: the source image, which should be a grayscale image,
    55: the threshold value which is used to classify the pixel values,
    255: the maximum value which is assigned to pixel values exceeding the threshold,
    cv2.CHAIN_APPROX_NONE:  the contour-approximation method
"""
ret, gray = cv2.threshold(blur, 55, 255, cv2.CHAIN_APPROX_NONE)
# save the "gray" image to ./resources/jpeg
cv2.imwrite('./resources/jpeg/treesImageBinary.png', gray)

#creating the structuring element, an ellipse
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#Morphological opening
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
# save the "opening" image to ./resources/jpeg
cv2.imwrite('./resources/jpeg/treesImageOpening.png', opening)

"""
Canny Edge Detection is used to detect the edges in an image. 
It accepts a gray scale image as input and it uses a multistage algorithm. 
Syntax: Canny(image, edges, threshold1, threshold2)
    image − A Mat object representing the source (input image) for this operation.
    edges − A Mat object representing the destination (edges) for this operation.
    threshold1 − A variable of the type double representing the first threshold for the hysteresis procedure.
    threshold2 − A variable of the type double representing the second threshold for the hysteresis procedure.
"""
edged = cv2.Canny(opening, 30, 200)
cv2.imwrite('./resources/jpeg/treesImageCanny.png', edged)

#creating the structuring element, an ellipse
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#Morphological closing
closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# save the "closing" image to ./resources/jpeg
cv2.imwrite('./resources/jpeg/treesImageClosing.png', closing)

"""
find the contours with the function findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    closing.copy() : source image, 
    cv2.RETR_EXTERNAL : contour retrieval mode, 
    cv2.CHAIN_APPROX_NONE : contour approximation method
"""
(cnt, heirarchy) = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#convert an image from BGR color to RBG
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Draw boundaries
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
cv2.imwrite('./resources/jpeg/treesImageDrawContours.png', rgb)

#visualize the result : the number of the trees in the image
#print('Nombre d\'arbres dans l\'image est : ', len(cnt))




#------For PNG image------#

#reading the image
image = cv2.imread('./resources/treesImage.png')

#convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# save the "gray" image to ./resources/jpeg
cv2.imwrite('./resources/png/treesImageGray2.png', gray)

"""
GaussianBlur(src, ksize, sigmaX)
src: input image
ksize: Gaussian Kernel Size. [height width]. 
       height and width should be odd and can have different values. 
       If ksize is set to [0 0], then ksize is computed from sigma values.
sigmaX: Kernel standard deviation along X-axis (horizontal direction).
"""
blur = cv2.GaussianBlur(gray, (11, 11), 0)
# save the "blur" image to ./resources/jpeg
cv2.imwrite('./resources/png/treesImageBlur2.png', blur)

"""
thresholding the image with the threshold(blur, 55, 255, cv2.CHAIN_APPROX_NONE) function
    blur: the source image, which should be a grayscale image,
    55: the threshold value which is used to classify the pixel values,
    255: the maximum value which is assigned to pixel values exceeding the threshold,
    cv2.CHAIN_APPROX_NONE:  the contour-approximation method
"""
ret, gray = cv2.threshold(blur, 80, 255, cv2.CHAIN_APPROX_NONE)
# save the "gray" image to ./resources/jpeg
cv2.imwrite('./resources/png/treesImageBinary2.png', gray)

#creating the structuring element, an ellipse
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#Morphological closing
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
# save the "closing" image to ./resources/jpeg
cv2.imwrite('./resources/png/treesImageClosing2.png', closing)

"""
find the contours with the function findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    closing.copy() : source image, 
    cv2.RETR_EXTERNAL : contour retrieval mode, 
    cv2.CHAIN_APPROX_NONE : contour approximation method
"""
(cnt, heirarchy) = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#convert an image from BGR color to RBG
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Draw boundaries
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
cv2.imwrite('./resources/png/treesImageDrawContours2.png', rgb)

#visualize the result : the number of the trees in the image
print('Nombre d\'arbres dans l\'image est : ', len(cnt))
