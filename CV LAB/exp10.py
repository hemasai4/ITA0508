import cv2
import numpy as np
img = cv2.imread("image.jpg")
if img is not None:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    moved = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("Original Image", img)
    cv2.imshow("Moved Image", moved)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found!")