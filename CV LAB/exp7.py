
import cv2
import time

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Unable to access webcam!")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Store captured frames
frames = []

print("Webcam Started Successfully!")
print("Recording video for 10 seconds...")

# Capture video for 10 seconds
start_time = time.time()

while time.time() - start_time < 10:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # Store frame
    frames.append(frame.copy())

    # Display live video
    cv2.imshow("Webcam Video Capture", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()

if len(frames) == 0:
    print("No frames captured!")
    exit()

print("Video Captured Successfully!")

# Function to play captured video
def play_video(title, delay, step=1):

    for i in range(0, len(frames), step):

        frame = frames[i].copy()

        cv2.putText(
            frame,
            title,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Video Playback", frame)

        # Press Q to stop
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Normal speed playback
print("Playing Normal Speed...")
play_video("Normal Speed", 33)

# Slow motion playback
print("Playing Slow Motion...")
play_video("Slow Motion - 0.5x", 66)

# Fast motion playback
print("Playing Fast Motion...")
play_video("Fast Motion - 2x", 33, step=2)

print("Video Processing Completed!")
