import cv2
import numpy as np
img = cv2.imread("image.jpg")
if img is not None:
    h, w = img.shape[:2]
    pts1 = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])
    pts2 = np.float32([
        [0.1 * w, 0.1 * h],
        [0.9 * w, 0.05 * h],
        [0.95 * w, 0.9 * h],
        [0.05 * w, 0.85 * h]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, M, (w, h))
    cv2.imshow("Original Image", img)
    cv2.imshow("Perspective Transformation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found!")