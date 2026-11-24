
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

# Define Sobel X-axis gradient mask
kernel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Define Sobel Y-axis gradient mask
kernel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# Apply gradient masks
grad_x = cv2.filter2D(
    gray.astype(np.float32),
    cv2.CV_32F,
    kernel_x
)

grad_y = cv2.filter2D(
    gray.astype(np.float32),
    cv2.CV_32F,
    kernel_y
)

# Calculate gradient magnitude
gradient = cv2.magnitude(grad_x, grad_y)

# Convert gradient to uint8
gradient = np.clip(
    gradient, 0, 255
).astype(np.uint8)

# Sharpen the original image
sharpened = cv2.addWeighted(
    gray, 1.0,
    gradient, 0.5,
    0
)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gradient, cmap="gray")
plt.title("Gradient Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap="gray")
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()

# Save output
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Gradient_Masking_Output.png"
)

plt.savefig(output_path)

# Show images
plt.show()

print("Gradient Masking Sharpening Successful!")
print("Output saved as Gradient_Masking_Output.png")
