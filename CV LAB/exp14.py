
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Read input image
image_path = "image.jpg"

if not os.path.exists(image_path):
    print("Error: image.jpg not found!")
    exit()

img = cv2.imread(image_path)

if img is None:
    print("Error: Unable to read image!")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width = img.shape[:2]

# Define source points
src = np.float32([
    [0, 0],
    [width - 1, 0],
    [0, height - 1],
    [width - 1, height - 1]
])

# Define destination points
dst = np.float32([
    [50, 50],
    [350, 0],
    [0, 350],
    [400, 400]
])

# Calculate Homography Matrix
H = cv2.getPerspectiveTransform(src, dst)

# Apply perspective transformation
result = cv2.warpPerspective(img, H, (450, 450))

# Display original and transformed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Homography Transformed Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print Homography Matrix
print("Homography Matrix:")
print(H)

print("Homography Transformation Successful!")
