
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

# Convert original image to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(dilated, cmap="gray")
plt.title("Dilation Image")
plt.axis("off")

plt.tight_layout()

# Save output
cv2.imwrite(
    os.path.join(folder, "Dilation_Output.png"),
    dilated
)

plt.savefig(
    os.path.join(folder, "Experiment30_Output.png")
)

# Show images
plt.show()

print("Dilation Operation Successful!")
print("Output saved as Dilation_Output.png")
