import cv2
import mediapipe as mp
import time
import numpy as np
import random



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
    
threshold = 50
def generate_frames():
    # Initialize MediaPipe Pose model
   

    # Initialize variables
    keypoints_prev = None
    motion_start_time = 0
    strokes = 0
    speeds = []

    # Open webcam

    
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
                if angle > 200:
                    cv2.putText(frame, "Elbow and hands are positioned too high for paddling.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                elif angle < -150:
                    cv2.putText(frame, "Elbow and hands are positioned too low for paddling.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Elbow and hands are positioned correctly for paddling.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            
            left_elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            prev_time = time.time()
            if prev_time - motion_start_time >= 1:
                prev_keypoints = keypoints
                motion_start_time = prev_time
                prev_left_elbow = left_elbow
            
            prev_time = 0
            if left_elbow:
                cv2.putText(frame, "Elbow Position: ({}, {})".format(left_elbow[0], left_elbow[1]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
                cv2.putText(frame, "Previous Elbow Position: ({}, {})".format(prev_left_elbow[0], prev_left_elbow[1]), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            if prev_left_elbow:
                distance = np.linalg.norm(np.array(left_elbow) - np.array(prev_left_elbow))
                cv2.putText(frame, "Speed (in pixels/s): {:.2f}".format(distance), (80, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            prev_left_elbow = left_elbow
            # if prev_elbow:
                # current_elbow = left_elbow
                # distance = current_elbow[0] - prev_elbow[0], current_elbow[1] - prev_elbow[1]
                # cv2.putText(frame, "Distance: ({:.2f}, {:.2f})".format(distance[0], distance[1]), (80, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            # prev_elbow = left_elbow
           
            # Display results
            avg_speed = random.uniform(0, 10)  # Random average speed between 0 and 20
            strokes_per_second = random.uniform(0, 2)  # Random strokes per second between 0 and 5

            # cv2.putText(frame, "Average Speed: {:.2f} pixels/second".format(avg_speed), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            # cv2.putText(frame, "Strokes per second: {:.2f}".format(strokes_per_second), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            if len(speeds) > 0:  # Check if speeds list is not empty before accessing speed
                cv2.putText(frame, "Arm Movement Speed: {:.2f} pixels/second".format(speed), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
                if detect_consistency(speeds):
                    cv2.putText(frame, "Movement Speed Consistent", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Movement Speed Inconsistent", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

            prev_elbow = None
            elbow_positions = []
            # Check if the elbow has moved significantly
            if len(keypoints) > 0:
                elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value] if keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value] else keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                if elbow:
                    # Calculate the distance between the current elbow position and the previous elbow position
                    if prev_elbow:
                        distance = euclidean_distance(elbow, prev_elbow)
                        # If the distance is greater than a threshold, count it as a stroke
                        if distance > 20:
                            stroke_count += 1
                    # Update the previous elbow position
                    # cv2.putText(frame, "Distance: {:.2f}".format(distance), (80, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    elbow_positions.append(elbow)
                    while len(elbow_positions) > 2:
                        distance = ((elbow_positions[-1][0] - elbow_positions[-2][0]) ** 2 + (elbow_positions[-1][1] - elbow_positions[-2][1]) ** 2) ** 0.5
                        cv2.putText(frame, "Distance: {:.2f}".format(distance), (70, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                    # print(distance)
                    # print(elbow)
                    # print(prev_elbow)
                    # Calculate the distance between consecutive elbow positions
                        # distance = euclidean_distance(elbow, prev_elbow)
                        # print(distance)
                        # Print the distance to the screen
                    # Update the previous elbow position
                                        # prev_elbow = elbow

                    prev_elbow = elbow
                  


            # Display stroke count and strokes per second on the frame
            # cv2.putText(frame, "Stroke Count: {}".format(stroke_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
            # cv2.putText(frame, "Strokes per Second: {:.2f}".format(strokes_per_second), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
    
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    cap.release()
    cv2.destroyAllWindows()

generate_frames()
# Load the object segmentation model