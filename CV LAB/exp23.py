
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

# Convert image to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(original, (9, 9), 10)

# Create Unsharp Mask
unsharp_mask = cv2.subtract(original, blurred)

# Sharpen the original image
sharpened = cv2.addWeighted(
    original, 1.5,
    blurred, -0.5,
    0
)

# Display images
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(blurred)
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(unsharp_mask)
plt.title("Unsharp Mask")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(sharpened)
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()

# Save output image
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Unsharp_Masking_Output.png"
)

plt.savefig(output_path)

# Show images
plt.show()

print("Unsharp Masking Successfully Completed!")
print("Output saved as Unsharp_Masking_Output.png")
