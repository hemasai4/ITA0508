
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

# Convert grayscale image to binary
_, binary = cv2.threshold(
    gray, 127, 255,
    cv2.THRESH_BINARY
)

# Define structuring element
kernel = np.ones((5, 5), np.uint8)

# Apply erosion operation
eroded = cv2.erode(
    binary,
    kernel,
    iterations=1
)

# Convert original image to RGB
original = cv2.cvtColor(
    img, cv2.COLOR_BGR2RGB
)

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
plt.imshow(eroded, cmap="gray")
plt.title("Erosion Image")
plt.axis("off")

plt.tight_layout()

# Save output images
folder = os.path.dirname(os.path.abspath(__file__))

cv2.imwrite(
    os.path.join(folder, "Erosion_Output.png"),
    eroded
)

plt.savefig(
    os.path.join(folder, "Experiment29_Output.png")
)

# Show output
plt.show()

print("Erosion Operation Successful!")
print("Output saved as Erosion_Output.png")

