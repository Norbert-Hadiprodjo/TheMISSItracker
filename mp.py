import cv2
import mediapipe as mp

# Load MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Access webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make detection
            results = pose.process(frame_rgb)

            # Draw landmarks and connections
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                          landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

            cv2.imshow('Pose Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

generate_frames()
