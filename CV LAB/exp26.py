
import cv2
import os
import matplotlib.pyplot as plt

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

# Create a copy of original image
watermarked = img.copy()

# Get image dimensions
height, width = img.shape[:2]

# Watermark text
text = "WATERMARK"

# Set font properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = max(0.5, width / 800)
thickness = 2

# Calculate text size
text_size = cv2.getTextSize(
    text, font, font_scale, thickness
)[0]

# Set watermark position at center
x = (width - text_size[0]) // 2
y = (height + text_size[1]) // 2

# Create overlay
overlay = img.copy()

# Add watermark text
cv2.putText(
    overlay,
    text,
    (x, y),
    font,
    font_scale,
    (255, 255, 255),
    thickness,
    cv2.LINE_AA
)

# Apply transparency
alpha = 0.4

watermarked = cv2.addWeighted(
    overlay, alpha,
    img, 1 - alpha,
    0
)

# Convert images to RGB
original_rgb = cv2.cvtColor(
    img, cv2.COLOR_BGR2RGB
)

watermarked_rgb = cv2.cvtColor(
    watermarked, cv2.COLOR_BGR2RGB
)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(watermarked_rgb)
plt.title("Watermarked Image")
plt.axis("off")

plt.tight_layout()

# Save watermarked image
output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Watermarked_Output.png"
)

cv2.imwrite(output_path, watermarked)

# Show output
plt.show()

print("Watermark Inserted Successfully!")
print("Output saved as Watermarked_Output.png")
