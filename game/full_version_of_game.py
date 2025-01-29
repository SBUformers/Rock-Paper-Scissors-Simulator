import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import cv2
import time
from multiprocessing import Process, Queue, Event
from PIL import Image, ImageTk
import torch
import numpy as np
import dlib
from imutils import face_utils
import mediapipe as mp
import pygame

# Initialize pygame mixer
pygame.mixer.init()
# Load the sound
countdown_end_sound = "boxing-bell.mp3"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mask_image = cv2.imread('red_mask.png', cv2.IMREAD_UNCHANGED)
if mask_image is None:
    print("Error: Mask image could not be loaded.")
    exit()
    
crown_img = cv2.imread("crown.png", cv2.IMREAD_UNCHANGED)
# crown_img = cv2.resize(crown_img, (100, 100), interpolation=cv2.INTER_LINEAR)
if crown_img is None:
    print("Error: Crown image could not be loaded.")
    exit()


def capture_frames(queue, stop_event):
    """Captures frames from the webcam and puts them in the queue."""
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally
            if not queue.full():
                queue.put(frame)
    cap.release()

def rotate_image(image, angle):
    """Rotates an image around its center."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def overlay_image(background, overlay, x, y, scale=1.0, angle=0):
    """
    Overlay one image (overlay) on another (background) at position (x, y) with a scale and rotation angle.
    """
    # Scale the overlay image
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape

    # Rotate the overlay image
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_overlay = cv2.warpAffine(overlay, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Define the region of interest (ROI) on the background
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    overlay_y1, overlay_y2 = max(0, -y), min(h, background.shape[0] - y)
    overlay_x1, overlay_x2 = max(0, -x), min(w, background.shape[1] - x)

    # Extract the alpha channel from the overlay
    alpha = rotated_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    alpha_inv = 1.0 - alpha

    # Blend the overlay with the background
    for c in range(3):  # Iterate over RGB channels
        background[y1:y2, x1:x2, c] = (
            alpha * rotated_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
            alpha_inv * background[y1:y2, x1:x2, c]
        )

    return background

def overlay_mask(frame, face_rect):
    """Overlays the red mask on the detected face."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert (fx, fy, fw, fh) to dlib.rectangle
    if isinstance(face_rect, tuple):
        fx, fy, fw, fh = face_rect
        face_rect = dlib.rectangle(fx, fy, fx + fw, fy + fh)

    shape = predictor(gray, face_rect)  # Now it's in the correct format
    shape = face_utils.shape_to_np(shape)

    # Extract key landmarks
    chin_point = shape[8]  # Chin
    under_eye_left = shape[36]  # Left eye
    under_eye_right = shape[45]  # Right eye

    # Calculate angle for mask rotation
    delta_y = under_eye_right[1] - under_eye_left[1]
    delta_x = under_eye_right[0] - under_eye_left[0]
    angle = -np.degrees(np.arctan2(delta_y, delta_x))

    # Calculate mask dimensions
    mask_width = int(np.linalg.norm(shape[16] - shape[0]) * 1.1)
    mask_height = int(np.linalg.norm(chin_point - ((under_eye_left[0] + under_eye_right[0]) // 2, 
                                                   (under_eye_left[1] + under_eye_right[1]) // 2)) * 1.4)

    # Rotate and resize mask
    rotated_mask = rotate_image(mask_image, angle)
    resized_mask = cv2.resize(rotated_mask, (mask_width, mask_height))

    # Calculate mask placement
    center_x = (chin_point[0] + under_eye_left[0] + under_eye_right[0]) // 3
    center_y = chin_point[1] - (mask_height // 2) + 10

    # Define the region of interest (ROI)
    x1, y1 = max(0, center_x - mask_width // 2), max(0, center_y - mask_height // 2)
    x2, y2 = min(frame.shape[1], center_x + mask_width // 2), min(frame.shape[0], center_y + mask_height // 2)
    roi = frame[y1:y2, x1:x2]

    # Overlay mask on frame
    if resized_mask.shape[2] == 4:
        alpha_mask = resized_mask[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_mask

        alpha_mask = alpha_mask[:roi.shape[0], :roi.shape[1]]
        alpha_frame = alpha_frame[:roi.shape[0], :roi.shape[1]]
        resized_mask = resized_mask[:roi.shape[0], :roi.shape[1]]

        for c in range(0, 3):
            roi[:, :, c] = (alpha_mask * resized_mask[:, :, c] +
                            alpha_frame * roi[:, :, c])

        frame[y1:y2, x1:x2] = roi

def overlay_crown(frame):
    """Detect faces and overlay a crown using MediaPipe FaceMesh."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, c = frame.shape

            # Get landmarks for forehead and chin
            forehead_x = int(face_landmarks.landmark[10].x * w)
            forehead_y = int(face_landmarks.landmark[10].y * h)
            chin_x = int(face_landmarks.landmark[152].x * w)
            chin_y = int(face_landmarks.landmark[152].y * h)

            # Calculate center point for the crown (slightly above forehead)
            crown_center_x = forehead_x
            crown_center_y = forehead_y - int((chin_y - forehead_y) * 1.2)

            # Ensure scale is always positive
            scale = max(0.0, (chin_y - forehead_y) / crown_img.shape[0] * 1.5)

            # Calculate rotation angle based on the face tilt
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            delta_y = (right_eye_outer.y - left_eye_outer.y) * h
            delta_x = (right_eye_outer.x - left_eye_outer.x) * w
            angle = -np.degrees(np.arctan2(delta_y, delta_x))

            # Overlay the crown on the frame
            frame = overlay_image(frame, crown_img, crown_center_x - int(scale * crown_img.shape[1] // 2), crown_center_y, scale, angle)

    return frame

def zoom_on_face(frame, face_rect, zoom_factor=1.5):
    """Zooms in on the winner's face while keeping the frame size the same."""
    fx, fy, fw, fh = face_rect
    cx, cy = fx + fw // 2, fy + fh // 2  # Face center

    # Calculate new crop box
    new_w = int(fw * zoom_factor)
    new_h = int(fh * zoom_factor)
    x1, y1 = max(0, cx - new_w // 2), max(0, cy - new_h // 2)
    x2, y2 = min(frame.shape[1], cx + new_w // 2), min(frame.shape[0], cy + new_h // 2)

    # Crop & resize back to original frame size
    zoomed_face = frame[y1:y2, x1:x2]
    zoomed_face = cv2.resize(zoomed_face, (frame.shape[1], frame.shape[0]))

    return zoomed_face


def process_frames(queue, model, stop_event, victory_condition, gui_queue):
    """
    Processes frames from the queue using the YOLO model.
    Modified to start the countdown and set boundary lines after 1 second of detecting two fists.
    """
    player_left_score = 0
    player_right_score = 0
    round_active = False
    countdown_active = False
    countdown_start_time = 0
    delay_active = False
    delay_start_time = 0
    winner_message = None
    guidance_message = "Show 'rock' to start the round!"
    round_end_time = 0  # Track when the round ends
    round_end_delay = 3  # Delay in seconds before the next round can start
    winner_detected = None

    # Store the single boundary line (y-value) for each player
    target_positions = {"PlayerLeft": None, "PlayerRight": None}

    # Track cheating
    cheat_flags = {"PlayerLeft": False, "PlayerRight": False}
    cheat_reasons = {"PlayerLeft": "", "PlayerRight": ""}
    
    # For each player, track whether they've shown "outside" and "inside"
    # within each 2-second segment of the 4-second countdown.
    touch_history = {
        "PlayerLeft":  {"outside": False, "inside": False},
        "PlayerRight": {"outside": False, "inside": False}
    }

    # Timing parameters
    countdown_duration = 4   # Countdown in seconds
    check_interval = 4       # Check crossing every 2 seconds
    last_check_time = 0      # Track the last time we performed the 2-second check
    padding = 20             # Distance to offset the line from the player's initial y_min

    # New variables for gesture change detection
    gesture_check_active = False
    gesture_check_start_time = 0
    initial_gestures = {"PlayerLeft": None, "PlayerRight": None}
    initial_gestures_captured = False  # Track if initial gestures have been captured

    # Variable to track when two fists are detected
    fists_detected_time = None

    # Track which player cheated in the current round
    cheated_players = {"PlayerLeft": False, "PlayerRight": False}

    # # # TODO: Debug
    # player_left_score = victory_condition  # Force left player to meet the victory condition
    # winner_detected = "PlayerLeft"  # Manually set the winner

    while not stop_event.is_set():
        current_time = time.time()

        # Process frames if available
        if not queue.empty():
            frame = queue.get()

            if winner_detected:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                zoom_factor = 1.25  # Initial zoom level (1.0 = no zoom)
                target_zoom = 2  # Target zoom level
                zoom_speed = 0.05  # Speed of zoom adjustment
                crown_vertical_offset = 10  # Adjust this value to lower the crown further
                margin_ratio = 1.2  # Extra margin around the face and crown

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        h, w, c = frame.shape

                        # Get landmarks for forehead and chin
                        forehead_x = int(face_landmarks.landmark[10].x * w)
                        forehead_y = int(face_landmarks.landmark[10].y * h)
                        chin_x = int(face_landmarks.landmark[152].x * w)
                        chin_y = int(face_landmarks.landmark[152].y * h)

                        face_center_x = forehead_x  # X-coordinate of the face center

                        # Check if the winner is PlayerLeft or PlayerRight
                        if (winner_detected == "PlayerLeft" and face_center_x < frame.shape[1] // 2) or \
                        (winner_detected == "PlayerRight" and face_center_x >= frame.shape[1] // 2):

                            # Calculate center point for the zoom (midpoint between forehead and chin)
                            center_x = (forehead_x + chin_x) // 2
                            center_y = (forehead_y + chin_y) // 2

                            # Determine the bounding box around the face and crown with margin
                            face_width = int((chin_y - forehead_y) * margin_ratio)
                            face_height = int((chin_y - forehead_y) * margin_ratio)

                            # Smoothly adjust the zoom factor toward the target zoom
                            if zoom_factor < target_zoom:
                                zoom_factor += zoom_speed
                            elif zoom_factor > target_zoom:
                                zoom_factor -= zoom_speed

                            # Calculate the cropped area based on the zoom factor
                            zoom_w = int(w / zoom_factor)
                            zoom_h = int(h / zoom_factor)
                            top_left_x = max(center_x - zoom_w // 2, 0)
                            top_left_y = max(center_y - zoom_h // 2, 0)
                            bottom_right_x = min(center_x + zoom_w // 2, w)
                            bottom_right_y = min(center_y + zoom_h // 2, h)

                            # Ensure the cropping region stays within bounds
                            cropped_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                            # Resize the cropped frame back to the original frame size
                            frame = cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_LINEAR)

                            # Adjust crown's position relative to the zoomed frame
                            crown_center_x = int((forehead_x - top_left_x) * (w / (bottom_right_x - top_left_x)))
                            crown_center_y = int((forehead_y - top_left_y) * (h / (bottom_right_y - top_left_y))) - int((chin_y - forehead_y) * 1.2 * (h / (bottom_right_y - top_left_y)))

                            # Adjust scale for the crown relative to the zoomed frame
                            scale = (chin_y - forehead_y) / crown_img.shape[0] * 1.5 * (h / (bottom_right_y - top_left_y))

                            # Calculate rotation angle based on the face tilt
                            left_eye_outer = face_landmarks.landmark[33]
                            right_eye_outer = face_landmarks.landmark[263]
                            delta_y = (right_eye_outer.y - left_eye_outer.y) * h
                            delta_x = (right_eye_outer.x - left_eye_outer.x) * w
                            angle = -np.degrees(np.arctan2(delta_y, delta_x))

                            # Overlay the crown on the zoomed frame
                            frame = overlay_image(
                                frame,
                                crown_img,
                                crown_center_x - int(scale * crown_img.shape[1] // 2),
                                crown_center_y,
                                scale,
                                angle
                            )

                            break  # Only process the first detected winner




            # Get predictions from the YOLO model
            results = model(frame)
            result = results[0]
            predictions = result.boxes.data

            # Filter predictions to only use the two with the highest confidence scores
            if len(predictions) > 2:
                # Extract confidence scores and sort predictions
                predictions = predictions[predictions[:, 4].argsort(descending=True)]
                predictions = predictions[:2]

            # Associate faces with the closest hands
            face_assignments = {}  # Store each face's closest hand
            hands = []

            # Extract hand positions first (use x_min, y_min)
            if len(predictions) >= 2:
                for box in predictions:
                    x_min, y_min, x_max, y_max, conf, cls = box.tolist()
                    label = model.names[int(cls)].lower()
                    hands.append((x_min, y_min, label))

            # for player in ["PlayerLeft", "PlayerRight"]:
            #     if cheat_flags[player]:
            #         cheated_players[player] = True  # Mark as cheated
            
            if any(cheated_players.values()):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                for face in faces:
                    # Unpack face coordinates based on type (tuple or dlib rectangle)
                    if isinstance(face, tuple):
                        fx, fy, fw, fh = face
                    else:
                        fx, fy, fw, fh = face.left(), face.top(), face.width(), face.height()

                    # Determine which side of the screen the face belongs to
                    face_center_x = fx + (fw // 2)
                    player_side = "PlayerLeft" if face_center_x < frame.shape[1] // 2 else "PlayerRight"

                    # Apply mask only if the player is marked as cheating
                    if cheated_players[player_side]:
                        overlay_mask(frame, (fx, fy, fw, fh))  # Cover the cheating player's face

            # Initialize placeholders for left/right gestures
            player_left_gesture = None
            player_right_gesture = None
            player_left_y_min, player_right_y_min = None, None
            player_left_y_max, player_right_y_max = None, None


            # We need at least 2 hands detected (one for each player)
            if len(predictions) >= 2:
                hands = []
                for box in predictions:
                    x_min, y_min, x_max, y_max, conf, cls = box.tolist()
                    label = model.names[int(cls)].lower()
                    hands.append((x_min, label, y_min, y_max))

                # Sort by x_min to determine who is Left vs. Right
                hands.sort(key=lambda h: h[0])
                (player_left_x,  player_left_gesture,  player_left_y_min,  player_left_y_max)  = hands[0]
                (player_right_x, player_right_gesture, player_right_y_min, player_right_y_max) = hands[1]

                # Check if both players are showing 'rock' to start the round
                if not countdown_active and not round_active:
                    if player_left_gesture == "rock" and player_right_gesture == "rock":
                        if fists_detected_time is None:
                            fists_detected_time = current_time
                        elif current_time - fists_detected_time >= 1:
                            # Ensure that the round has ended and the delay has passed
                            if current_time - round_end_time >= round_end_delay:
                                countdown_active = True
                                round_active = True
                                countdown_start_time = time.time()
                                guidance_message = "Round starts! Move your hand around the line!"

                                # Set the single boundary line for each player (just below y_min)
                                target_positions["PlayerLeft"]  = player_left_y_min  - padding
                                target_positions["PlayerRight"] = player_right_y_min - padding

                                # Reset cheating/tracking
                                cheat_flags = {"PlayerLeft": False, "PlayerRight": False}
                                cheat_reasons = {"PlayerLeft": "", "PlayerRight": ""}
                                touch_history = {
                                    "PlayerLeft":  {"outside": False, "inside": False},
                                    "PlayerRight": {"outside": False, "inside": False}
                                }
                                last_check_time = time.time()
                                fists_detected_time = None

                                # Reset cheated players for the new round
                                cheated_players = {"PlayerLeft": False, "PlayerRight": False}

                # If we already had a winner message, let them show 'rock' again to reset
                if winner_message and not countdown_active and not round_active:
                    if player_left_gesture == "rock" and player_right_gesture == "rock":
                        winner_message = None
                        guidance_message = "Show 'rock' to start the round!"

            # Countdown logic
            if countdown_active:
                elapsed_time = current_time - countdown_start_time
                countdown_remaining = max(0, countdown_duration - int(elapsed_time))

                # As long as we have valid y_min for each player, update inside/outside
                if player_left_y_min is not None:
                    line_left = target_positions["PlayerLeft"]
                    if player_left_y_min < line_left:
                        touch_history["PlayerLeft"]["outside"] = True
                    else:
                        touch_history["PlayerLeft"]["inside"] = True

                if player_right_y_min is not None:
                    line_right = target_positions["PlayerRight"]
                    if player_right_y_min < line_right:
                        touch_history["PlayerRight"]["outside"] = True
                    else:
                        touch_history["PlayerRight"]["inside"] = True

                # Every 2 seconds, check if the player has gone both outside and inside
                if (current_time - last_check_time) >= check_interval:
                    for player in ["PlayerLeft", "PlayerRight"]:
                        # If they have NOT shown both states in this interval, mark them as cheating
                        if not (touch_history[player]["outside"] and touch_history[player]["inside"]):
                            cheat_flags[player] = True
                            cheat_reasons[player] = (
                                f"{player} didn't move around the line (both above/below) "
                                f"in {int(check_interval)} seconds!"
                            )
                            cheated_players[player] = True  # Mark the player as cheated
                        # Reset for the next interval
                        touch_history[player] = {"outside": False, "inside": False}
                    last_check_time = current_time

                # Once 4 seconds of countdown is done, move to the 2-second delay
                if countdown_remaining == 0:
                    pygame.mixer.Sound(countdown_end_sound).play()
                    countdown_active = False
                    delay_active = True
                    delay_start_time = time.time()
                    gesture_check_active = True
                    gesture_check_start_time = time.time()
                    initial_gestures["PlayerLeft"] = player_left_gesture
                    initial_gestures["PlayerRight"] = player_right_gesture
                    initial_gestures_captured = False  # Reset for the next round

            # Draw the single boundary lines (if round is active)
            if round_active:
                if target_positions["PlayerLeft"] is not None and player_left_x is not None:
                    cv2.line(
                        frame,
                        (int(player_left_x - 50),  int(target_positions["PlayerLeft"])),
                        (int(player_left_x + 50),  int(target_positions["PlayerLeft"])),
                        (0, 255, 0), 2
                    )
                if target_positions["PlayerRight"] is not None and player_right_x is not None:
                    cv2.line(
                        frame,
                        (int(player_right_x - 50), int(target_positions["PlayerRight"])),
                        (int(player_right_x + 50), int(target_positions["PlayerRight"])),
                        (0, 255, 0), 2
                    )

            # # After the countdown, we wait 2 seconds, then finalize the round
            # if delay_active:
            #     elapsed_delay = current_time - delay_start_time
                
            #     # 1) First second after countdown: "Perform your final move now!"
            #     if elapsed_delay < 1:
            #         guidance_message = "Perform your final move now!"
            #         # Here you let players adjust gestures freely
            #         # (no cheating is flagged during this 1-second window)
            #         # Keep track of the updated gestures if you'd like:
            #         initial_gestures["PlayerLeft"] = player_left_gesture
            #         initial_gestures["PlayerRight"] = player_right_gesture
        
            #     if elapsed_delay >= 2:
            #         delay_active = False

            #         # Check cheating
            #         both_cheated = cheat_flags["PlayerLeft"] and cheat_flags["PlayerRight"]
            #         left_cheated = cheat_flags["PlayerLeft"] and not cheat_flags["PlayerRight"]
            #         right_cheated = cheat_flags["PlayerRight"] and not cheat_flags["PlayerLeft"]

            #         if both_cheated:
            #             winner_message = "Both players cheated! No one wins this round."
            #             player_left_score -= 1
            #             player_right_score -= 1
            #         elif left_cheated:
            #             winner_message = f"Player Left cheated! {cheat_reasons['PlayerLeft']} No one wins this round."
            #             player_left_score -= 1
            #         elif right_cheated:
            #             winner_message = f"Player Right cheated! {cheat_reasons['PlayerRight']} No one wins this round."
            #             player_right_score -= 1
            #         else:
            #             # Determine the winner if neither cheated
            #             if len(predictions) >= 2 and player_left_gesture and player_right_gesture:
            #                 if player_left_gesture == player_right_gesture:
            #                     winner_message = "It's a tie!"
            #                 elif (
            #                     (player_left_gesture == "rock"     and player_right_gesture == "scissors") or
            #                     (player_left_gesture == "scissors" and player_right_gesture == "paper")    or
            #                     (player_left_gesture == "paper"    and player_right_gesture == "rock")
            #                 ):
            #                     player_left_score += 1
            #                     winner_message = "Player Left wins the round!"
            #                 else:
            #                     player_right_score += 1
            #                     winner_message = "Player Right wins the round!"

            #                 # Check victory condition
            #                 if player_left_score == victory_condition:
            #                     winner_message = "Player Left wins the game!"
            #                     stop_event.set()
            #                 elif player_right_score == victory_condition:
            #                     winner_message = "Player Right wins the game!"
            #                     stop_event.set()
            #             else:
            #                 winner_message = "Not enough players detected to determine a winner."

            #         # Reset everything for the next round
            #         round_active = False
            #         cheat_flags = {"PlayerLeft": False, "PlayerRight": False}
            #         cheat_reasons = {"PlayerLeft": "", "PlayerRight": ""}
            #         round_end_time = current_time  # Record the time the round ended
            
            if delay_active:
                elapsed_delay = current_time - delay_start_time

                # 1) First second after countdown: "Perform your final move now!"
                if elapsed_delay < 2:
                    guidance_message = "Perform your final move now!"
                    # Here you let players adjust gestures freely
                    # (no cheating is flagged during this 1-second window)
                    # Keep track of the updated gestures if you'd like:
                    initial_gestures["PlayerLeft"] = player_left_gesture
                    initial_gestures["PlayerRight"] = player_right_gesture

                # 2) Second second after countdown: "Freeze! No changes allowed!"
                elif 2 <= elapsed_delay < 4:
                    guidance_message = "Freeze! No changes allowed!"
                    # If they change the gesture in this window, mark as cheating
                    if player_left_gesture and initial_gestures["PlayerLeft"]:
                        if player_left_gesture != initial_gestures["PlayerLeft"]:
                            cheat_flags["PlayerLeft"] = True
                            cheat_reasons["PlayerLeft"] = "Player Left changed gesture during Freeze!"

                    if player_right_gesture and initial_gestures["PlayerRight"]:
                        if player_right_gesture != initial_gestures["PlayerRight"]:
                            cheat_flags["PlayerRight"] = True
                            cheat_reasons["PlayerRight"] = "Player Right changed gesture during Freeze!"

                # After 2 total seconds, finalize the round
                else:
                    delay_active = False

                    # -- Check cheating and decide the winner exactly as before --
                    both_cheated = cheat_flags["PlayerLeft"] and cheat_flags["PlayerRight"]
                    left_cheated = cheat_flags["PlayerLeft"] and not cheat_flags["PlayerRight"]
                    right_cheated = cheat_flags["PlayerRight"] and not cheat_flags["PlayerLeft"]

                    if both_cheated:
                        winner_message = "Both players cheated! No one wins this round."
                        player_left_score -= 1
                        player_right_score -= 1
                    elif left_cheated:
                        winner_message = f"Player Left cheated! {cheat_reasons['PlayerLeft']} No one wins this round."
                        player_left_score -= 1
                    elif right_cheated:
                        winner_message = f"Player Right cheated! {cheat_reasons['PlayerRight']} No one wins this round."
                        player_right_score -= 1
                    else:
                        # Determine winner if neither cheated
                        if len(predictions) >= 2 and player_left_gesture and player_right_gesture:
                            if player_left_gesture == player_right_gesture:
                                winner_message = "It's a tie!"
                            elif (
                                (player_left_gesture == "rock"     and player_right_gesture == "scissors") or
                                (player_left_gesture == "scissors" and player_right_gesture == "paper")    or
                                (player_left_gesture == "paper"    and player_right_gesture == "rock")
                            ):
                                player_left_score += 1
                                winner_message = "Player Left wins the round!"
                            else:
                                player_right_score += 1
                                winner_message = "Player Right wins the round!"

                            # Check victory condition
                            if player_left_score == victory_condition:
                                winner_message = "Player Left wins the game!"
                                winner_detected = "PlayerLeft"

                            elif player_right_score == victory_condition:
                                winner_message = "Player Right wins the game!"
                                winner_detected = "PlayerRight"
                        else:
                            winner_message = "Not enough players detected to determine a winner."

                    # Reset everything for the next round
                    round_active = False
                    cheat_flags = {"PlayerLeft": False, "PlayerRight": False}
                    cheat_reasons = {"PlayerLeft": "", "PlayerRight": ""}
                    round_end_time = current_time  # Record the time the round ended

            # Check for gesture changes between 0.5 and 1.5 seconds after countdown
            if gesture_check_active:
                elapsed_gesture_check = current_time - gesture_check_start_time
                if elapsed_gesture_check == 3:
                    if player_left_gesture != initial_gestures["PlayerLeft"]:
                        cheat_flags["PlayerLeft"] = True
                        cheat_reasons["PlayerLeft"] = "Player Left changed their gesture during the forbidden window!"
                        cheated_players["PlayerLeft"] = True  # Mark the player as cheated
                    if player_right_gesture != initial_gestures["PlayerRight"]:
                        cheat_flags["PlayerRight"] = True
                        cheat_reasons["PlayerRight"] = "Player Right changed their gesture during the forbidden window!"
                        cheated_players["PlayerRight"] = True  # Mark the player as cheated
                elif elapsed_gesture_check > 1.5:
                    gesture_check_active = False

            # Capture initial gestures 1 second after countdown finishes
            if delay_active and not initial_gestures_captured:
                elapsed_delay = current_time - delay_start_time
                
                if elapsed_delay >= 1:
                    initial_gestures["PlayerLeft"] = player_left_gesture
                    initial_gestures["PlayerRight"] = player_right_gesture
                    initial_gestures_captured = True

            # Prepare GUI text
            info_text = f"Player Left: {player_left_score}   Player Right: {player_right_score}\n"
            countdown_text = ""
            if countdown_active:
                elapsed_time = int(time.time() - countdown_start_time)
                countdown_remaining = max(0, countdown_duration - elapsed_time)
                countdown_text = f"Countdown: {countdown_remaining}"

            guidance_text = guidance_message if guidance_message else ""
            winner_text = winner_message if winner_message else ""

            # Send text info to the GUI
            gui_queue.put((info_text, countdown_text, guidance_text, winner_text))

            # Display annotated frame
            annotated_frame = result.plot()
            combined_frame = cv2.addWeighted(annotated_frame, 0.7, frame, 0.3, 0)
            gui_queue.put(combined_frame)

            if cv2.waitKey(1) == 27:  # Stop on 'Esc' key
                stop_event.set()
                break

    cv2.destroyAllWindows()



class RockPaperScissorsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rock Paper Scissors Game")
        self.root.geometry("800x700")  # Set window size

        # Custom style for the UI
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Helvetica", 14), padding=10)
        self.style.configure("TButton", font=("Helvetica", 12), padding=10)
        self.style.configure("TEntry", font=("Helvetica", 12), padding=10)

        # Frame for the scoreboard
        self.score_frame = ttk.Frame(root)
        self.score_frame.pack(fill=tk.X, padx=10, pady=10)

        # Labels for scores
        self.player_left_label = ttk.Label(self.score_frame, text="Player Left: 0", foreground="blue")
        self.player_left_label.pack(side=tk.LEFT, padx=10)

        self.player_right_label = ttk.Label(self.score_frame, text="Player Right: 0", foreground="green")
        self.player_right_label.pack(side=tk.RIGHT, padx=10)

        # Label for countdown
        self.countdown_label = ttk.Label(root, text="", foreground="orange")
        self.countdown_label.pack(fill=tk.X, padx=10, pady=5)

        # Label for guidance messages
        self.guidance_label = ttk.Label(root, text="Welcome! Press 'Start Game' to begin.", foreground="purple")
        self.guidance_label.pack(fill=tk.X, padx=10, pady=5)

        # Label for winner messages
        self.winner_label = ttk.Label(root, text="", foreground="red")
        self.winner_label.pack(fill=tk.X, padx=10, pady=5)

        # Entry for number of victories
        self.victory_condition_label = ttk.Label(root, text="Enter the number of victories needed to win:")
        self.victory_condition_label.pack(pady=10)

        self.victory_condition_entry = ttk.Entry(root)
        self.victory_condition_entry.pack(pady=10)

        # Start button
        self.start_button = ttk.Button(root, text="Start Game", command=self.start_game)
        self.start_button.pack(pady=20)

        # Canvas for video feed (with no padding)
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack(pady=0, padx=0)  # No padding around the canvas

        # Queue for communication between processes
        self.gui_queue = Queue()

    def start_game(self):
        victory_condition = int(self.victory_condition_entry.get())
        self.victory_condition_label.pack_forget()
        self.victory_condition_entry.pack_forget()
        self.start_button.pack_forget()

        # Load the YOLO model
        # model = YOLO('yolo11-rps-detection.pt')
        model = YOLO('fineTuned_best_V2.pt')

        # Check if CUDA is available and move the model to the GPU
        if torch.cuda.is_available():
            print('cuda is available')
            model.to('cuda')

        # Create shared resources
        frame_queue = Queue(maxsize=5)
        stop_event = Event()

        # Start the processes
        capture_process = Process(target=capture_frames, args=(frame_queue, stop_event))
        process_process = Process(target=process_frames, args=(frame_queue, model, stop_event, victory_condition, self.gui_queue))

        capture_process.start()
        process_process.start()

        # Start updating the GUI
        self.update_gui()

    def update_gui(self):
        if not self.gui_queue.empty():
            item = self.gui_queue.get()
            if isinstance(item, tuple):
                # Update the labels with the new text
                info_text, countdown_text, guidance_text, winner_text = item
                self.player_left_label.config(text=info_text.split("   ")[0])
                self.player_right_label.config(text=info_text.split("   ")[1])
                self.countdown_label.config(text=countdown_text)
                self.guidance_label.config(text=guidance_text)
                self.winner_label.config(text=winner_text)
            else:
                # Update the canvas with the new frame
                frame = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.image = imgtk

        self.root.after(10, self.update_gui)

if __name__ == "__main__":
    root = tk.Tk()
    app = RockPaperScissorsApp(root)
    root.mainloop()