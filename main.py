import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./resources/treesImage.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite('./resources/treesImageGray.png',gray)
blur = cv2.GaussianBlur(gray ,(11,11),0)
cv2.imwrite('./resources/treesImageBlur.png',blur)
canny = cv2.Canny(blur,30,150,3)
cv2.imwrite('./resources/treesImageCanny.png',canny)
dilated = cv2.dilate(canny,(1,1),iterations=2)
cv2.imwrite('./resources/treesImageDilate.png',dilated)

(cnt,heirarchy) = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb,cnt,-1,(0,255,0),2)
cv2.imwrite('./resources/treesImageDrawContours.png',rgb)
print('Nombre d\'arbres dans l\'image est : ',len(cnt))