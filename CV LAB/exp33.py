
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Read input image
folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, "image.jpg")

if not os.path.exists(image_path):
    print("Error: image.jpg not found!")
    exit()

img = cv2.imread(image_path)

if img is None:
    print("Error: Unable to read image!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert grayscale image into binary
_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

# Create a 5x5 structuring element
kernel = np.ones((5, 5), np.uint8)

# Apply dilation operation
dilated = cv2.dilate(
    binary, kernel, iterations=1
)

# Apply erosion operation
eroded = cv2.erode(
    binary, kernel, iterations=1
)

# Apply Morphological Gradient
gradient = cv2.morphologyEx(
    binary,
    cv2.MORPH_GRADIENT,
    kernel
)

# Convert original image to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(dilated, cmap="gray")
plt.title("Dilated Image")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(eroded, cmap="gray")
plt.title("Eroded Image")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(gradient, cmap="gray")
plt.title("Morphological Gradient")
plt.axis("off")

plt.tight_layout()

# Save output images
cv2.imwrite(
    os.path.join(folder, "Gradient_Output.png"),
    gradient
)

plt.savefig(
    os.path.join(folder, "Experiment33_Output.png")
)

# Show images
plt.show()

print("Morphological Gradient Operation Successful!")
print("Output saved as Gradient_Output.png")
