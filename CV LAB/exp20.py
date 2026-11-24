
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

# Laplacian mask with negative center coefficient
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

# Apply Laplacian filter
laplacian = cv2.filter2D(
    gray.astype(np.float32),
    cv2.CV_32F,
    kernel
)

# Sharpen image by subtracting Laplacian
sharpened = gray.astype(np.float32) - laplacian

# Convert output to uint8
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Convert Laplacian for display
laplacian_display = cv2.convertScaleAbs(laplacian)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(laplacian_display, cmap="gray")
plt.title("Laplacian Filter")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap="gray")
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()

# Save output
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Laplacian_Sharpened_Output.png"
)

plt.savefig(output_path)

# Show output
plt.show()

print("Laplacian Image Sharpening Successful!")
print("Output saved as Laplacian_Sharpened_Output.png")
