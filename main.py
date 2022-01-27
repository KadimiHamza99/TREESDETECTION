import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Load Image
image = cv2.imread('./resources/treesImage.jpeg')
cv2.waitKey(0)
# blur the image to smmooth out the edges a bit, also reduces a bit of noise
# blurred = cv2.GaussianBlur(image, (5, 5), 0)
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# apply thresholding to conver the image to binary format
# after this operation all the pixels below 200 value will be 0...
# and all th pixels above 200 will be 255
ret, gray = cv2.threshold(gray, 60, 255, cv2.CHAIN_APPROX_NONE)
cv2.imwrite('./resources/treesImageGray.png', gray)
kernel = np.ones((3, 3), 'uint8')
dilated = cv2.dilate(gray, kernel, iterations=1)

cv2.imwrite('./resources/treesImageDilate.png', dilated)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('./resources/treesImageOpening.png', opening)
cv2.imwrite('./resources/treesImageClosing.png', closing)
(cnt, heirarchy) = cv2.findContours(
    opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
cv2.imwrite('./resources/treesImageDrawContours.png', rgb)
print('Nombre d\'arbres dans l\'image est : ', len(cnt))


closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
cv2.imwrite('./resources/treesImageClosing.png', closing)
edged = cv2.Canny(opening, 70, 255)
cv2.imwrite('./resources/treesImageCanny.png', edged)
"""

image = cv2.imread('./resources/treesImage.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./resources/treesImageGray.png', gray)
blur = cv2.GaussianBlur(gray, (11, 11), 0)
cv2.imwrite('./resources/treesImageBlur.png', blur)
ret, gray = cv2.threshold(blur, 80, 255, cv2.CHAIN_APPROX_NONE)
cv2.imwrite('./resources/treesImageBinary.png', gray)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('./resources/treesImageClosing.png', closing)

(cnt, heirarchy) = cv2.findContours(
    closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
cv2.imwrite('./resources/treesImageDrawContours.png', rgb)
print('Nombre d\'arbres dans l\'image est : ', len(cnt))
