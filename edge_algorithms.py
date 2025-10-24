import cv2
import numpy as np

def apply_canny(image, lower_thresh, upper_thresh, ksize, sigma):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    edges = cv2.Canny(blur, lower_thresh, upper_thresh)
    return edges

def apply_sobel(image, ksize, direction):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = 1 if direction in ("x", "both") else 0
    dy = 1 if direction in ("y", "both") else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
    return cv2.convertScaleAbs(sobel)

def apply_laplacian(image, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(lap)
