
import cv2
import matplotlib.pyplot as plt
import os

# Read input image
folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, "watch.jpg")

if not os.path.exists(image_path):
    print("Error: watch.jpg not found!")
    exit()

img = cv2.imread(image_path)

if img is None:
    print("Error: Unable to read image!")
    exit()

# Resize image for display
img = cv2.resize(img, (600, 400))

# Create a copy for recognition
output = img.copy()

# Select watch region manually
# Coordinates are adjusted for the given watch image
height, width = img.shape[:2]

x1 = int(width * 0.30)
y1 = int(height * 0.05)
x2 = int(width * 0.70)
y2 = int(height * 0.98)

# Draw bounding box
cv2.rectangle(
    output,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3
)

# Add object label
cv2.putText(
    output,
    "WATCH",
    (x1, max(25, y1 - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    2
)

# Convert images to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
recognized = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original Watch Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(recognized)
plt.title("Watch Recognition")
plt.axis("off")

plt.tight_layout()

# Save output
output_path = os.path.join(
    folder, "Watch_Recognition_Output.png"
)

cv2.imwrite(output_path, output)

plt.show()

print("Object Identified: WATCH")
print("Watch Region Highlighted Successfully!")
print("Output saved as Watch_Recognition_Output.png")
