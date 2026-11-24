
import cv2
import os

# Get current folder
folder = os.path.dirname(os.path.abspath(__file__))

# Input video path
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

# Get video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# Calculate frame delay
normal_delay = max(1, int(1000 / fps))
slow_delay = max(1, int(2000 / fps))
fast_delay = max(1, int(500 / fps))

# Function to play video
def play_video(title, delay):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Playing:", title)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 480))

        # Display speed label
        cv2.putText(
            frame,
            title,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display video
        cv2.imshow("Video Processing", frame)

        # Press Q to quit
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            return False

    return True

# Play normal speed
if play_video("Normal Speed", normal_delay):

    # Play slow motion
    if play_video("Slow Motion - 0.5x", slow_delay):

        # Play fast motion
        play_video("Fast Motion - 2x", fast_delay)

# Release video
cap.release()
cv2.destroyAllWindows()

print("Video Processing Completed!")
