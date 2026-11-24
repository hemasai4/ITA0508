
import cv2
import os

# Get current folder
folder = os.path.dirname(os.path.abspath(__file__))

# Video file path
video_path = os.path.join(folder, "video.mp4")

# Check video file
if not os.path.exists(video_path):
    print("Error: video.mp4 not found!")
    exit()

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video!")
    exit()

# Create background subtractor
background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

# Create morphological kernel
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (5, 5)
)

print("Vehicle Detection Started...")
print("Press Q to exit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize video frame
    frame = cv2.resize(frame, (800, 500))

    # Apply background subtraction
    mask = background.apply(frame)

    # Remove noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Apply dilation
    mask = cv2.dilate(
        mask,
        kernel,
        iterations=2
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_count = 0

    # Draw bounding boxes
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 1500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Filter small objects
        if w < 40 or h < 30:
            continue

        vehicle_count += 1

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display label
        cv2.putText(
            frame,
            "Moving Object",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display moving object count
    cv2.putText(
        frame,
        "Detected Objects: " + str(vehicle_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display video frames
    cv2.imshow("Vehicle Detection", frame)
    cv2.imshow("Motion Mask", mask)

    # Press Q to stop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Vehicle Detection Completed!")
