# Save the face of the user in encoded form using Fuzzy Vault
import time
import os
import sys
import json
import configparser
import builtins
import numpy as np
from recorders.video_capture import VideoCapture
# Try to import dlib and give a nice error if we can't
try:
    import dlib
except ImportError as err:
    print(err)
    print("Can't import the dlib module, check the output of")
    print("pip3 show dlib")
    sys.exit(1)

# OpenCV needs to be imported after dlib
import cv2
from vault_utils import deterministic_secret_from_biometric, create_vault_from_coeffs


# Read config from disk
config = configparser.ConfigParser()
config.read("/usr/local/etc/bm_auth/face_auth/config.ini")

use_cnn = config.getboolean("core", "use_cnn", fallback=False)
if use_cnn:
    face_detector = dlib.cnn_face_detection_model_v1("/usr/local/share/dlib-data/mmod_human_face_detector.dat")
else:
    face_detector = dlib.get_frontal_face_detector()

pose_predictor = dlib.shape_predictor("/usr/local/share/dlib-data/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("/usr/local/share/dlib-data/dlib_face_recognition_resnet_model_v1.dat")

user = builtins.bm_user
enc_file = f"/usr/local/etc/bm_auth/face_auth/models/{user}.dat"
encodings = []

# Create models folder if needed
if not os.path.exists("/usr/local/etc/bm_auth/face_auth/models"):
    print("No face model folder found, creating one")
    os.makedirs("/usr/local/etc/bm_auth/face_auth/models")

# Load existing encodings
try:
    encodings = json.load(open(enc_file))
except FileNotFoundError:
    encodings = []

# Warning for too many models
if len(encodings) > 9:
    print("NOTICE: Each additional model slows down recognition")
    print("Press Ctrl+C to cancel\n")

# User prompt setup
print("Adding face model for the user " + user)

# Get base label
base_label = "Composite Model"

# Clean label
if "," in base_label:
    print("NOTICE: Removing commas from model name")
    base_label = base_label.replace(",", "")

# Directions setup
directions = [
    {"suffix": "Front", "message": "Please look straight into the camera"},
    {"suffix": "Left", "message": "Please turn head to the LEFT"},
    {"suffix": "Right", "message": "Please turn head to the RIGHT"}
]

# Get next available ID
next_id = encodings[-1]["id"] + 1 if encodings else 0
group_number = next_id // 3

# Main capture loop
for idx, direction in enumerate(directions):
    print(f"\n{direction['message']}")
    time.sleep(3)

    # Capture setup
    video_capture = VideoCapture(config)
    frames = 0
    valid_frames = 0
    dark_tries = 0
    dark_running_total = 0
    face_locations = None

    # Frame processing loop
    while frames < 60:
        frames += 1
        frame, gsframe = video_capture.read_frame()
        gsframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gsframe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gsframe)

        # Darkness calculation
        hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
        hist_total = np.sum(hist)
        if hist_total == 0:
            continue

        darkness = (hist[0] / hist_total * 100)
        dark_running_total += darkness
        valid_frames += 1

        if darkness > config.getfloat("video", "dark_threshold", fallback=60):
            dark_tries += 1
            continue

        # Face detection
        face_locations = face_detector(gsframe, 1)
        if face_locations:
            break

    # Error handling
    if not face_locations:
        print("No face detected for direction: " + direction["suffix"])
        sys.exit(1)

    if len(face_locations) > 1:
        print("Multiple faces detected")
        sys.exit(1)

    # Face encoding
    face_location = face_locations[0].rect if use_cnn else face_locations[0]
    face_landmark = pose_predictor(frame, face_location)
    face_encoding = np.array(face_encoder.compute_face_descriptor(frame, face_landmark, 1))
    face_encoding /= np.linalg.norm(face_encoding)

    # Create fuzzy Vault
    local_coeffs = deterministic_secret_from_biometric(face_encoding)
    vault = create_vault_from_coeffs(local_coeffs, face_encoding.tolist())

    model_label = f"{base_label} #{group_number} ({direction['suffix']})"
    model_id = next_id + idx

    encodings.append({
        "time": int(time.time()),
        "label": model_label,
        "id": model_id,
        "vault": vault
    })

    video_capture.release()

# Save model
with open(enc_file, "w") as datafile:
    json.dump(encodings, datafile)

print("Added 3 models (Group #{}) to {}".format(group_number, user))
