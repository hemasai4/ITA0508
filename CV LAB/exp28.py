
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

# Define convolution kernel
kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

# Apply convolution filter
boundary = cv2.filter2D(
    blur,
    cv2.CV_32F,
    kernel
)

# Convert to displayable image
boundary = cv2.convertScaleAbs(boundary)

# Apply threshold for clear boundaries
_, boundary_binary = cv2.threshold(
    boundary, 30, 255,
    cv2.THRESH_BINARY
)

# Convert original image to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(boundary, cmap="gray")
plt.title("Convolution Boundary")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(boundary_binary, cmap="gray")
plt.title("Detected Boundary")
plt.axis("off")

plt.tight_layout()

# Save output
folder = os.path.dirname(os.path.abspath(__file__))

cv2.imwrite(
    os.path.join(folder, "Boundary_Output.png"),
    boundary_binary
)

plt.savefig(
    os.path.join(folder, "Experiment28_Output.png")
)

# Show output
plt.show()

print("Boundary Detection Successful!")
print("Output saved as Boundary_Output.png")
