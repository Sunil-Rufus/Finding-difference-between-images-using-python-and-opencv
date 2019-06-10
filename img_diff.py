from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os

imageA = cv2.imread("C:/Users/SUNIL RUFUS/Desktop/pyth/hevid4/int0.jpg")
imageB = cv2.imread("C:/Users/SUNIL RUFUS/Desktop/pyth/hevid4/int301.jpg")

scale_percent = 30
widtha = int(imageA.shape[1]*scale_percent/100)
heighta = int(imageA.shape[0]*scale_percent/100)
widthb = int(imageB.shape[1]*scale_percent/100)
heightb = int(imageB.shape[0]*scale_percent/100)
dima = (widtha,heighta)
dimb = (widthb,heightb)

imageA = cv2.resize(imageA,dima)
imageB = cv2.resize(imageB,dimb)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

grayA = cv2.resize(grayA,dima)
grayB = cv2.resize(grayB,dimb)
#grayA = cv2.resize(imageA, (100, 100))
#grayB = cv2.resize(imageB, (200, 100))
# convert the images to grayscale

print(grayA.shape)
print(grayB.shape)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
