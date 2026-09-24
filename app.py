from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from feature_extractor import get_knee_angle
from rep_counter import RepCounter


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# MediaPipe Pose setup
# --------------------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# Camera
# --------------------------------------------------

camera = cv2.VideoCapture(0)


# --------------------------------------------------
# Repetition Counter
# --------------------------------------------------

rep_counter = RepCounter(
    up_threshold=160,
    down_threshold=100
)


# --------------------------------------------------
# Generate video frames
# --------------------------------------------------

def generate_frames():

    while True:

        # Read frame from camera
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame from camera.")
            break

        # --------------------------------------------------
        # Convert BGR → RGB
        # --------------------------------------------------

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------
        # Process frame using MediaPipe
        # --------------------------------------------------

        results = pose.process(image)

        # Convert RGB → BGR for OpenCV display
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # --------------------------------------------------
        # Check if pose was detected
        # --------------------------------------------------

        if results.pose_landmarks:

            # Draw skeleton
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            try:

                # Get body landmarks
                landmarks = results.pose_landmarks.landmark

                # Calculate knee angle
                knee_angle, is_reliable = get_knee_angle(
                    landmarks,
                    mp_pose,
                    side="left"
                )

                # --------------------------------------------------
                # Reliable knee angle
                # --------------------------------------------------

                if is_reliable:

                    # Update repetition counter
                    rep_counter.update(knee_angle)

                    # Display knee angle
                    cv2.putText(
                        image,
                        f"Angle: {int(knee_angle)}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

                    # Display exercise stage
                    cv2.putText(
                        image,
                        f"Stage: {rep_counter.get_stage()}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

                # --------------------------------------------------
                # Knee angle not reliable
                # --------------------------------------------------

                else:

                    cv2.putText(
                        image,
                        "Legs not visible",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            except Exception as e:

                # Prevent one bad frame from crashing the application
                print("Feature extraction error:", e)

        else:

            # No human pose detected
            cv2.putText(
                image,
                "No person detected",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # --------------------------------------------------
        # Display repetition count
        # --------------------------------------------------

        cv2.putText(
            image,
            f"Reps: {rep_counter.get_count()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # --------------------------------------------------
        # Convert frame to JPEG
        # --------------------------------------------------

        ret, buffer = cv2.imencode(
            ".jpg",
            image
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # --------------------------------------------------
        # Send frame to browser
        # --------------------------------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def index():

    return render_template("index.html")


# --------------------------------------------------
# Video stream
# --------------------------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )