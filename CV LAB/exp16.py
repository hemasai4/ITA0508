
import cv2
import matplotlib.pyplot as plt
import os

# Read input image
image_path = "image.jpg"

if not os.path.exists(image_path):
    print("Error: image.jpg not found!")
    exit()

img = cv2.imread(image_path)

if img is None:
    print("Error: Unable to read image!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

# Apply Canny Edge Detection
edges = cv2.Canny(blur, 100, 200)

# Convert original image to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.tight_layout()

# Save output
plt.savefig("Canny_Output.png")

# Show output
plt.show()

print("Canny Edge Detection Successful!")
print("Output saved as Canny_Output.png")
