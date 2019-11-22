import cv2
import numpy as np

cap = cv2.VideoCapture(0)
im = cap.read()[1]
print im.shape
cap.release()
