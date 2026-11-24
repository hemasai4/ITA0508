
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

# Create a 9x9 structuring element
kernel = np.ones((9, 9), np.uint8)

# Apply Closing operation
closing = cv2.morphologyEx(
    gray,
    cv2.MORPH_CLOSE,
    kernel
)

# Apply Black Hat operation
blackhat = cv2.morphologyEx(
    gray,
    cv2.MORPH_BLACKHAT,
    kernel
)

# Convert original image to RGB
original = cv2.cvtColor(
    img, cv2.COLOR_BGR2RGB
)

# Display images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(closing, cmap="gray")
plt.title("Closing Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(blackhat, cmap="gray")
plt.title("Black Hat Image")
plt.axis("off")

plt.tight_layout()

# Save output images
cv2.imwrite(
    os.path.join(folder, "BlackHat_Output.png"),
    blackhat
)

plt.savefig(
    os.path.join(folder, "Experiment35_Output.png")
)

# Show images
plt.show()

print("Black Hat Operation Successful!")
print("Output saved as BlackHat_Output.png")
