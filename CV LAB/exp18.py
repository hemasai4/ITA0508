
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Read input image
image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "image.jpg"
)

if not os.path.exists(image_path):
    print("Error: image.jpg not found!")
    exit()

img = cv2.imread(image_path)

if img is None:
    print("Error: Unable to read image!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Sobel operator along Y-axis
sobel_y = cv2.Sobel(
    blur,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

# Convert to displayable image
sobel_y = cv2.convertScaleAbs(sobel_y)

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
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y Edge Detection")
plt.axis("off")

plt.tight_layout()

# Save output
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Sobel_Y_Output.png"
)
plt.savefig(output_path)

# Show output
plt.show()

print("Sobel Y-axis Edge Detection Successful!")
print("Output saved as Sobel_Y_Output.png")
