import cv2
import mediapipe as mp
import time
import numpy as np

# Function to calculate the Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2) ** 0.5

# Function to calculate speed of movement
def calculate_speed(keypoints1, keypoints2, time_diff):
    total_distance = 0
    for kp1, kp2 in zip(keypoints1, keypoints2):
        total_distance += euclidean_distance(kp1, kp2)
    return total_distance / time_diff if time_diff > 0 else 0

# Function to detect motion consistency
def detect_consistency(speeds, threshold=0.5):
    if len(speeds) < 2:
        return True
    else:
        avg_speed = sum(speeds) / len(speeds)
        for speed in speeds:
            if abs(speed - avg_speed) > threshold:
                return False
        return True


def generate_frames():
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize variables
    keypoints_prev = None
    motion_start_time = None
    strokes = 0
    speeds = []

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)

            # Extract keypoints
            keypoints = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    keypoints.append((x, y))

            if len(keypoints) > 0:
                if keypoints_prev is not None:
                    time_diff = time.time() - motion_start_time if motion_start_time else 0
                    speed = calculate_speed(keypoints_prev, keypoints, time_diff)
                    speeds.append(speed)
                    if detect_consistency(speeds):
                        cv2.putText(frame, "Consistent Movement", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Inconsistent Movement", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    strokes += 1

                keypoints_prev = keypoints[:]
                motion_start_time = time.time() if motion_start_time is None else motion_start_time

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

generate_frames()


