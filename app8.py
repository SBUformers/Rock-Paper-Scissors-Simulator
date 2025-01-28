import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import cv2
import time
from multiprocessing import Process, Queue, Event
from PIL import Image, ImageTk
import torch
import numpy as np

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



def process_frames(queue, model, stop_event, victory_condition, gui_queue):
    """
    Processes frames from the queue using the YOLO model.
    Modified to use only one (upper) boundary line for each player.
    A player is marked as cheating if, in every 2-second interval of the 4-second countdown,
    they do not cross that line at least once (showing both outside and inside states).
    Additionally, a player is marked as cheating if they change their gesture between
    0.5 and 1.5 seconds after the countdown ends.
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

    while not stop_event.is_set():
        current_time = time.time()

        # Process frames if available
        if not queue.empty():
            frame = queue.get()
            
            # Get predictions from the YOLO model
            results = model(frame)
            result = results[0]
            predictions = result.boxes.data

            # Filter predictions to only use the two with the highest confidence scores
            if len(predictions) > 2:
                # Extract confidence scores and sort predictions
                predictions = predictions[predictions[:, 4].argsort(descending=True)]
                predictions = predictions[:2]

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
                        # Reset for the next interval
                        touch_history[player] = {"outside": False, "inside": False}
                    last_check_time = current_time

                # Once 4 seconds of countdown is done, move to the 2-second delay
                if countdown_remaining == 0:
                    countdown_active = False
                    delay_active = True
                    delay_start_time = time.time()
                    gesture_check_active = True
                    gesture_check_start_time = time.time()
                    initial_gestures["PlayerLeft"] = player_left_gesture
                    initial_gestures["PlayerRight"] = player_right_gesture

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

            # After the countdown, we wait 2 seconds, then finalize the round
            if delay_active:
                elapsed_delay = current_time - delay_start_time
                if elapsed_delay >= 2:
                    delay_active = False

                    # Check cheating
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
                        # Determine the winner if neither cheated
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
                                stop_event.set()
                            elif player_right_score == victory_condition:
                                winner_message = "Player Right wins the game!"
                                stop_event.set()
                        else:
                            winner_message = "Not enough players detected to determine a winner."


                    # Reset everything for the next round
                    round_active = False
                    cheat_flags = {"PlayerLeft": False, "PlayerRight": False}
                    cheat_reasons = {"PlayerLeft": "", "PlayerRight": ""}

            # Check for gesture changes between 0.5 and 1.5 seconds after countdown
            if gesture_check_active:
                elapsed_gesture_check = current_time - gesture_check_start_time
                if 3 <= elapsed_gesture_check <= 10:
                    if player_left_gesture != initial_gestures["PlayerLeft"]:
                        cheat_flags["PlayerLeft"] = True
                        cheat_reasons["PlayerLeft"] = "Player Left changed their gesture during the forbidden window!"
                    if player_right_gesture != initial_gestures["PlayerRight"]:
                        cheat_flags["PlayerRight"] = True
                        cheat_reasons["PlayerRight"] = "Player Right changed their gesture during the forbidden window!"
                elif elapsed_gesture_check > 1.5:
                    gesture_check_active = False

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
        model = YOLO('fineTuned_best_V1.pt')

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