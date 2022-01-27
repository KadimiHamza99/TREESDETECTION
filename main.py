import cv2
import numpy as np
import matplotlib.pyplot as plt















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
