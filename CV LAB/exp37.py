
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

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# Set playback speed
delay = max(1, int(1000 / fps))

print("Total Frames:", total_frames)
print("FPS:", fps)
print("Playing Video in Reverse Mode...")

# Play video from last frame to first frame
for frame_no in range(total_frames - 1, -1, -1):

    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    # Read frame
    ret, frame = cap.read()

    if not ret:
        continue

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # Display reverse video
    cv2.imshow("Reverse Video Playback", frame)

    # Press Q to stop video
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release video
cap.release()
cv2.destroyAllWindows()

print("Reverse Video Playback Completed!")
