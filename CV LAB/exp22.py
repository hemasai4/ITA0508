
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

# Laplacian sharpening mask
# Positive center coefficient
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

# Apply sharpening filter
sharpened = cv2.filter2D(
    gray,
    -1,
    kernel
)

# Display output images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sharpened, cmap="gray")
plt.title("Laplacian Sharpened Image")
plt.axis("off")

plt.tight_layout()

# Save output image
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Laplacian_Positive_Output.png"
)

plt.savefig(output_path)

# Show images
plt.show()

print("Laplacian Positive Center Sharpening Successful!")
print("Output saved as Laplacian_Positive_Output.png")
