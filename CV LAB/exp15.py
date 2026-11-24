
import cv2
import numpy as np
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

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width = img.shape[:2]

# Define source points
src = np.float64([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

# Define destination points
dst = np.float64([
    [50, 50],
    [350, 0],
    [400, 400],
    [0, 350]
])

# Direct Linear Transformation function
def compute_dlt(src, dst):
    A = []

    for (x, y), (u, v) in zip(src, dst):
        A.append([
            -x, -y, -1, 0, 0, 0,
            u*x, u*y, u
        ])

        A.append([
            0, 0, 0, -x, -y, -1,
            v*x, v*y, v
        ])

    A = np.array(A, dtype=np.float64)

    # Solve using Singular Value Decomposition
    _, _, Vt = np.linalg.svd(A, full_matrices=True)

    H = Vt[-1].reshape(3, 3)

    # Normalize matrix
    H = H / H[2, 2]

    return H

# Calculate transformation matrix
H = compute_dlt(src, dst)

# Apply DLT transformation
result = cv2.warpPerspective(img, H, (450, 450))

# Display output images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("DLT Transformed Image")
plt.axis("off")

plt.tight_layout()

# Save output image
plt.savefig("DLT_Output.png")

plt.show()

# Print transformation matrix
print("Direct Linear Transformation Matrix:")
print(H)

print("DLT Transformation Successful!")
