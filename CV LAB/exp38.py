
import cv2
import matplotlib.pyplot as plt
import os

# Get current folder
folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, "image.jpg")

# Check image file
if not os.path.exists(image_path):
    print("Error: image.jpg not found!")
    exit()

# Read input image
img = cv2.imread(image_path)

if img is None:
    print("Error: Unable to read image!")
    exit()

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Face classifier not loaded!")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Create output image
output = img.copy()

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        "Face",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# Convert images to RGB
original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
detected = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(detected)
plt.title("Face Detection")
plt.axis("off")

plt.tight_layout()

# Save output image
output_path = os.path.join(
    folder, "Face_Detection_Output.png"
)

cv2.imwrite(output_path, output)

# Show images
plt.show()

print("Face Detection Completed!")
print("Number of Faces Detected:", len(faces))
print("Output saved as Face_Detection_Output.png")
