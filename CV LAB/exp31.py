
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

# Apply erosion
eroded = cv2.erode(
    binary, kernel, iterations=1
)

# Apply Opening operation
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

# Convert original image to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(eroded, cmap="gray")
plt.title("Eroded Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(opening, cmap="gray")
plt.title("Opening Image")
plt.axis("off")

plt.tight_layout()

# Save output images
cv2.imwrite(
    os.path.join(folder, "Opening_Output.png"),
    opening
)

plt.savefig(
    os.path.join(folder, "Experiment31_Output.png")
)

# Show images
plt.show()

print("Opening Operation Successful!")
print("Output saved as Opening_Output.png")
