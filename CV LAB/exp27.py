
import cv2
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

# Resize original image
img = cv2.resize(img, (500, 400))

# Crop a portion of the image
cropped = img[50:200, 50:250].copy()

# Create another image for pasting
destination = img.copy()

# Copy and paste cropped image
destination[220:370, 280:480] = cropped

# Convert images to RGB
original_rgb = cv2.cvtColor(
    img, cv2.COLOR_BGR2RGB
)

cropped_rgb = cv2.cvtColor(
    cropped, cv2.COLOR_BGR2RGB
)

destination_rgb = cv2.cvtColor(
    destination, cv2.COLOR_BGR2RGB
)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cropped_rgb)
plt.title("Cropped Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(destination_rgb)
plt.title("Copied and Pasted Image")
plt.axis("off")

plt.tight_layout()

# Save output images
folder = os.path.dirname(os.path.abspath(__file__))

cv2.imwrite(
    os.path.join(folder, "Cropped_Output.png"),
    cropped
)

cv2.imwrite(
    os.path.join(folder, "Pasted_Output.png"),
    destination
)

plt.savefig(
    os.path.join(folder, "Experiment27_Output.png")
)

# Show output
plt.show()

print("Cropping, Copying and Pasting Successful!")
print("Output images saved successfully!")
