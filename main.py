import time

import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hand Landmark model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Function to calculate angle between three points
def calculate_angle(A, B, C):
    AB = np.array([B[0] - A[0], B[1] - A[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])

    cosine_angle = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


# Create a blank image with grids for each angle
def create_grid_image(angles, num_cols=3, circle_radius=50):
    if not angles:
        print("No angles to display.")
        return np.zeros((100, 100, 3), dtype=np.uint8)  # Return a blank image to avoid imshow error

    num_angles = len(angles)
    num_rows = math.ceil(num_angles / num_cols)
    image_height = num_rows * (2 * circle_radius + 20)
    image_width = num_cols * (2 * circle_radius + 20)
    grid_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for i, (name, angle) in enumerate(angles.items()):
        row = i // num_cols
        col = i % num_cols
        center_x = col * (2 * circle_radius + 20) + circle_radius + 10
        center_y = row * (2 * circle_radius + 20) + circle_radius + 10

        # Draw the circle
        cv2.circle(grid_image, (center_x, center_y), circle_radius, (255, 255, 255), -1)
        cv2.circle(grid_image, (center_x, center_y), circle_radius, (0, 0, 0), 2)

        # Put the angle text inside the circle
        cv2.putText(grid_image, f'{int(angle)}', (center_x - 15, center_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(grid_image, name, (center_x - circle_radius, center_y + circle_radius + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return grid_image


# Start capturing video input
cap = cv2.VideoCapture(0)
cur_time = time.time()

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process the image and detect hands
        # if time.time()-cur_time>0.5 or start_flag:

        cur_time=time.time()
        results = hands.process(image_rgb)
        print(time.time()-cur_time)
        start_flag=False
        # Draw the hand annotations on the image
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        angles = {}
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmarks and calculate angles
                landmarks = hand_landmarks.landmark

                # Index finger angles
                try:
                    angles['Index MCP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y],
                        [landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y],
                        [landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y]
                    )
                except:
                    angles['Index MCP'] = -1

                try:
                    angles['Index PIP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y],
                        [landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y],
                        [landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y]
                    )
                except:
                    angles['Index PIP'] = -1

                try:
                    angles['Index DIP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y],
                        [landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y],
                        [landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                         landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y]
                    )
                except:
                    angles['Index DIP'] = -1

                # Middle finger angles
                try:
                    angles['Middle MCP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y],
                        [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                         landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y],
                        [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y]
                    )
                except:
                    angles['Middle MCP'] = -1

                try:
                    angles['Middle PIP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                         landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y],
                        [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y],
                        [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
                         landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y]
                    )
                except:
                    angles['Middle PIP'] = -1

                try:
                    angles['Middle DIP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y],
                        [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
                         landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y],
                        [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                         landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y]
                    )
                except:
                    angles['Middle DIP'] = -1

                # Ring finger angles
                try:
                    angles['Ring MCP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y],
                        [landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                         landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y],
                        [landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y]
                    )
                except:
                    angles['Ring MCP'] = -1

                try:
                    angles['Ring PIP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                         landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y],
                        [landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y],
                        [landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x,
                         landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y]
                    )
                except:
                    angles['Ring PIP'] = -1

                try:
                    angles['Ring DIP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                         landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y],
                        [landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x,
                         landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y],
                        [landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                         landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y]
                    )
                except:
                    angles['Ring DIP'] = -1

                # Pinky finger angles
                try:
                    angles['Pinky MCP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y],
                        [landmarks[mp_hands.HandLandmark.PINKY_MCP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_MCP].y],
                        [landmarks[mp_hands.HandLandmark.PINKY_PIP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_PIP].y]
                    )
                except:
                    angles['Pinky MCP'] = -1

                try:
                    angles['Pinky PIP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.PINKY_MCP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_MCP].y],
                        [landmarks[mp_hands.HandLandmark.PINKY_PIP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_PIP].y],
                        [landmarks[mp_hands.HandLandmark.PINKY_DIP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_DIP].y]
                    )
                except:
                    angles['Pinky PIP'] = -1

                try:
                    angles['Pinky DIP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.PINKY_PIP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_PIP].y],
                        [landmarks[mp_hands.HandLandmark.PINKY_DIP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_DIP].y],
                        [landmarks[mp_hands.HandLandmark.PINKY_TIP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_TIP].y]
                    )
                except:
                    angles['Pinky DIP'] = -1

                # Thumb angles
                try:
                    angles['Thumb MCP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y],
                        [landmarks[mp_hands.HandLandmark.THUMB_CMC].x, landmarks[mp_hands.HandLandmark.THUMB_CMC].y],
                        [landmarks[mp_hands.HandLandmark.THUMB_MCP].x, landmarks[mp_hands.HandLandmark.THUMB_MCP].y]
                    )
                except:
                    angles['Thumb MCP'] = -1

                try:
                    angles['Thumb IP'] = calculate_angle(
                        [landmarks[mp_hands.HandLandmark.THUMB_CMC].x, landmarks[mp_hands.HandLandmark.THUMB_CMC].y],
                        [landmarks[mp_hands.HandLandmark.THUMB_MCP].x, landmarks[mp_hands.HandLandmark.THUMB_MCP].y],
                        [landmarks[mp_hands.HandLandmark.THUMB_IP].x, landmarks[mp_hands.HandLandmark.THUMB_IP].y]
                    )
                except:
                    angles['Thumb IP'] = -1

        # Create a grid image with angle values
        grid_image = create_grid_image(angles)

        # Display the resulting images
        cv2.imshow('Hand Tracking', image)
        cv2.imshow('Hand Angles', grid_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
