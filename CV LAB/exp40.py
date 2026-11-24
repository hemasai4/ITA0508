
import cv2
import os
import matplotlib.pyplot as plt

# Get current folder
folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, "image.jpg")

# Read input image
img = cv2.imread(image_path)

if img is None:
    print("Error: image.jpg not found!")
    exit()

# Resize image
img = cv2.resize(img, (600, 400))

# Create copy of original image
output = img.copy()

# Select rectangular regions
# Format: (x, y, width, height)
rectangles = [
    (155, 25, 130, 150),
    (290, 25, 130, 150)
]

# Draw rectangles and extract objects
for i, (x, y, w, h) in enumerate(rectangles):

    # Draw rectangle
    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    # Extract selected object
    cropped = img[y:y+h, x:x+w]

    # Save extracted object
    cv2.imwrite(
        os.path.join(folder, f"Object_{i+1}.png"),
        cropped
    )

    # Add label
    cv2.putText(
        output,
        f"Object {i+1}",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# Convert images to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Objects with Rectangles")
plt.axis("off")

plt.tight_layout()

# Save final output
cv2.imwrite(
    os.path.join(folder, "Experiment40_Output.png"),
    output
)

plt.show()

print("Rectangles Drawn Successfully!")
print("Objects Extracted Successfully!")
print("Output saved successfully!")
