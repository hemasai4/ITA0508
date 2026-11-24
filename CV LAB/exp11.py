import cv2
import numpy as np
img = cv2.imread("image.jpg")
if img is not None:
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    pts2 = np.float32([[0.1 * w, 0.15 * h],
                       [0.85 * w, 0.05 * h],
                       [0.2 * w, 0.9 * h]])
    M = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("Original Image", img)
    cv2.imshow("Affine Transformation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found!")