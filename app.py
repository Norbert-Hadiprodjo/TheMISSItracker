import cv2
import mediapipe as mp
import time
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)


# Function to calculate the Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2) ** 0.5

# Function to calculate speed of elbow movement
def calculate_speed(keypoints1, keypoints2, time_diff):
    elbow1 = keypoints1[mp_pose.PoseLandmark.LEFT_ELBOW.value] if keypoints1[mp_pose.PoseLandmark.LEFT_ELBOW.value] else keypoints1[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    elbow2 = keypoints2[mp_pose.PoseLandmark.LEFT_ELBOW.value] if keypoints2[mp_pose.PoseLandmark.LEFT_ELBOW.value] else keypoints2[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    distance = euclidean_distance(elbow1, elbow2)
    return distance / time_diff if time_diff > 0 else 0

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

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    keypoints_prev = None
    motion_start_time = 0
    strokes = 0
    speeds = []

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
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                          landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    keypoints.append((x, y))

            # Identify the elbow and hands
            if len(keypoints) > 0:
                elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value] if keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value] else keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_hand = keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_hand = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Provide feedback on positioning
            if elbow and left_hand and right_hand:
                # Calculate the angle between the elbow and hands
                angle = np.arctan2(right_hand[1] - elbow[1], right_hand[0] - elbow[0]) - np.arctan2(left_hand[1] - elbow[1], left_hand[0] - elbow[0])
                angle = np.degrees(angle)

                # Provide feedback based on the angle
                if angle > 120:
                    cv2.putText(frame, "Elbow and hands are positioned too high for paddling.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                elif angle < -120:
                    cv2.putText(frame, "Elbow and hands are positioned too low for paddling.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Elbow and hands are positioned correctly for paddling.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

            left_elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            prev_time = 0
            if left_elbow:
                # Draw a red circle around the left elbow
                height, width, _ = frame.shape
                cx, cy = int(left_elbow[0] * width), int(left_elbow[1] * height)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Calculate speed
                if prev_time != 0:
                    time_diff = time.time() - prev_time
                    displacement = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    speed = displacement / time_diff
                    cv2.imshow('elbow speed', speed)

                # Update previous position and time
                prev_cx, prev_cy = cx, cy
                prev_time = time.time()

            stroke_count = 0
            start_time = time.time()
            motion_start_time = 0

            strokes_per_second = 0
            if len(keypoints) > 0:
                if keypoints_prev is not None:
                    time_diff = time.time() - motion_start_time if motion_start_time else 0
                    speed = calculate_speed(keypoints_prev, keypoints, time_diff)
                    speeds.append(speed)
                    if detect_consistency(speeds):
                        if speed > 0 and motion_start_time == 0:
                            motion_start_time = time.time()
                        elif speed < 20 and motion_start_time > 0:
                            stroke_count += 1
                            motion_start_time = 0
                    else:
                        motion_start_time = 0
                    speeds = []
                avg_speed = sum(speeds) / len(speeds) if len(speeds) > 0 else 0
                strokes_per_second = stroke_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0

            prev_elbow = None
            if len(keypoints) > 0:
                elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value] if keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value] else keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                if elbow:
                    if prev_elbow:
                        distance = euclidean_distance(elbow, prev_elbow)
                        if distance > 20:
                            stroke_count += 1
                    prev_elbow = elbow
                    print(elbow)
                    print(stroke_count)

            cv2.putText(frame, "Stroke Count: {}".format(stroke_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            cv2.putText(frame, "Strokes per Second: {:.2f}".format(strokes_per_second), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
