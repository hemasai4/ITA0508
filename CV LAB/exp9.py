import cv2
img = cv2.imread("image.jpg")
if img is not None:
    clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    counterclockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow("Original Image", img)
    cv2.imshow("Clockwise Rotation", clockwise)
    cv2.imshow("Counterclockwise Rotation", counterclockwise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found!")