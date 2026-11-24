
import cv2
import matplotlib.pyplot as plt
import os

# Get current folder
folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, "image.jpg")

# Read input image
img = cv2.imread(image_path)

if img is None:
    print("Error: image.jpg not found!")
    exit()

# Get original dimensions
height, width = img.shape[:2]

# Scale image to bigger size (2x)
bigger = cv2.resize(
    img,
    None,
    fx=2.0,
    fy=2.0,
    interpolation=cv2.INTER_CUBIC
)

# Scale image to smaller size (0.5x)
smaller = cv2.resize(
    img,
    None,
    fx=0.5,
    fy=0.5,
    interpolation=cv2.INTER_AREA
)

# Convert BGR images to RGB
original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bigger_rgb = cv2.cvtColor(bigger, cv2.COLOR_BGR2RGB)
smaller_rgb = cv2.cvtColor(smaller, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(bigger_rgb)
plt.title("Bigger Image (2x)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(smaller_rgb)
plt.title("Smaller Image (0.5x)")
plt.axis("off")

plt.tight_layout()

# Save output images
cv2.imwrite(
    os.path.join(folder, "Bigger_Image.png"),
    bigger
)

cv2.imwrite(
    os.path.join(folder, "Smaller_Image.png"),
    smaller
)

plt.savefig(
    os.path.join(folder, "Experiment8_Output.png")
)

# Show images
plt.show()

print("Original Size:", width, "x", height)
print("Bigger Size:", bigger.shape[1], "x", bigger.shape[0])
print("Smaller Size:", smaller.shape[1], "x", smaller.shape[0])
print("Image Scaling Successful!")
