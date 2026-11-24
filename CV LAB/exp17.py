
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

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Sobel operator along X-axis
sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

# Convert to displayable image
sobel_x = cv2.convertScaleAbs(sobel_x)

# Convert original image to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X Edge Detection")
plt.axis("off")

plt.tight_layout()

# Save output
plt.savefig("Sobel_X_Output.png")

# Show output
plt.show()

print("Sobel X-axis Edge Detection Successful!")
print("Output saved as Sobel_X_Output.png")
