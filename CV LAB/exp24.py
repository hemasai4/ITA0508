
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

# High-boost amplification factor
A = 1.5

# High-boost mask with four neighbors
kernel1 = np.array([
    [0, -1, 0],
    [-1, A + 4, -1],
    [0, -1, 0]
], dtype=np.float32)

# High-boost mask with eight neighbors
kernel2 = np.array([
    [-1, -1, -1],
    [-1, A + 8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Apply high-boost filtering
boost4 = cv2.filter2D(
    gray, cv2.CV_32F, kernel1
)

boost8 = cv2.filter2D(
    gray, cv2.CV_32F, kernel2
)

# Normalize results to display range
boost4 = np.clip(boost4, 0, 255).astype(np.uint8)
boost8 = np.clip(boost8, 0, 255).astype(np.uint8)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(boost4, cmap="gray", vmin=0, vmax=255)
plt.title("High-Boost 4 Neighbors")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(boost8, cmap="gray", vmin=0, vmax=255)
plt.title("High-Boost 8 Neighbors")
plt.axis("off")

plt.tight_layout()

# Save output image
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "High_Boost_Output.png"
)

plt.savefig(output_path)

# Show output
plt.show()

print("High-Boost Image Sharpening Successful!")
print("Boost Factor A =", A)
print("Output saved as High_Boost_Output.png")
