import cv2
import numpy as np
import matplotlib.pyplot as plt

# For jpeg image
image = cv2.imread('./resources/treesImage.jpeg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./resources/jpeg/treesImageGray.png', gray)

blur = cv2.GaussianBlur(gray, (13, 13), 0)
cv2.imwrite('./resources/jpeg/treesImageBlur.png', blur)

ret, gray = cv2.threshold(blur, 55, 255, cv2.CHAIN_APPROX_NONE)
cv2.imwrite('./resources/jpeg/treesImageBinary.png', gray)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
cv2.imwrite('./resources/jpeg/treesImageOpening.png', opening)

edged = cv2.Canny(opening, 30, 200)
cv2.imwrite('./resources/jpeg/treesImageCanny.png', edged)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('./resources/jpeg/treesImageClosing.png', closing)

(cnt, heirarchy) = cv2.findContours(
    closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
cv2.imwrite('./resources/jpeg/treesImageDrawContours.png', rgb)

print('Nombre d\'arbres dans l\'image est : ', len(cnt))


# For PNG image
image = cv2.imread('./resources/treesImage.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./resources/png/treesImageGray.png', gray)

blur = cv2.GaussianBlur(gray, (11, 11), 0)
cv2.imwrite('./resources/png/treesImageBlur.png', blur)

ret, gray = cv2.threshold(blur, 80, 255, cv2.CHAIN_APPROX_NONE)
cv2.imwrite('./resources/png/treesImageBinary.png', gray)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('./resources/png/treesImageClosing.png', closing)

(cnt, heirarchy) = cv2.findContours(
    closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
cv2.imwrite('./resources/png/treesImageDrawContours.png', rgb)

print('Nombre d\'arbres dans l\'image est : ', len(cnt))