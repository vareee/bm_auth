# pam_face_auth.py — PAM совместимая версия аутентификации через Fuzzy Vault
import sys
import json
import time
import math
import numpy as np
import cv2
import dlib
import configparser
import random


# PAM interface
def pam_sm_authenticate(pamh, flags, argv):
    try:
        sys.path.append("/usr/local/lib/x86_64-linux-gnu/howdy")
        from recorders.video_capture import VideoCapture
        from vault_utils import unlock_vault

        user = pamh.get_user(None)
        if not user:
            return pamh.PAM_USER_UNKNOWN

        config = configparser.ConfigParser()
        config.read("/usr/local/etc/bm_auth/face_auth/config.ini")

        use_cnn = config.getboolean("core", "use_cnn", fallback=False)
        if use_cnn:
            face_detector = dlib.cnn_face_detection_model_v1("/usr/local/share/dlib-data/mmod_human_face_detector.dat")
        else:
            face_detector = dlib.get_frontal_face_detector()

        pose_predictor = dlib.shape_predictor("/usr/local/share/dlib-data/shape_predictor_5_face_landmarks.dat")
        face_encoder = dlib.face_recognition_model_v1("/usr/local/share/dlib-data/dlib_face_recognition_resnet_model_v1.dat")

        try:
            models = json.load(open(f"/usr/local/etc/bm_auth/face_auth/models/{user}.dat"))
        except FileNotFoundError:
            pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, "No face model found for user."))
            return pamh.PAM_AUTH_ERR

        available_directions = []
        for model in models:
            if "(Front)" in model["label"]:
                available_directions.append("Front")
            if "(Left)" in model["label"]:
                available_directions.append("Left")
            if "(Right)" in model["label"]:
                available_directions.append("Right")

        if not available_directions:
            pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, "No valid face directions found."))
            return pamh.PAM_AUTH_ERR

        selected_direction = random.choice(available_directions)
        direction_messages = {
            "Front": "Please look straight into the camera",
            "Left": "Please turn your head to the LEFT",
            "Right": "Please turn your head to the RIGHT"
        }

        pamh.conversation(pamh.Message(pamh.PAM_TEXT_INFO, direction_messages[selected_direction]))
        time.sleep(2)

        target_models = [m for m in models if f"({selected_direction})" in m["label"]]
        if not target_models:
            pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, f"No models for direction: {selected_direction}"))
            return pamh.PAM_AUTH_ERR

        def get_head_pose(landmarks):
            left_eye = np.array([landmarks.part(2).x, landmarks.part(2).y])
            right_eye = np.array([landmarks.part(0).x, landmarks.part(0).y])
            nose = np.array([landmarks.part(4).x, landmarks.part(4).y])
            eye_center = (left_eye + right_eye) / 2
            head_axis = nose - eye_center
            angle_rad = math.atan2(head_axis[1], head_axis[0])
            return math.degrees(angle_rad)

        def is_head_position_correct(angle_deg, expected):
            threshold = 15
            if expected == "Front":
                return 90 - threshold <= angle_deg <= 90 + threshold
            elif expected == "Left":
                return angle_deg < 90 - threshold
            elif expected == "Right":
                return angle_deg > 90 + threshold
            return False

        video_capture = VideoCapture(config)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        start_time = time.time()
        timeout = config.getint("video", "timeout", fallback=5)
        frame_id = 0

        while time.time() - start_time < timeout:
            frame, gsframe = video_capture.read_frame()
            if frame is None:
                continue

            frame_id += 1
            gsframe = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            face_locations = face_detector(gsframe, 1)

            for fl in face_locations:
                if use_cnn:
                    fl = fl.rect

                face_landmark = pose_predictor(frame, fl)
                angle_deg = get_head_pose(face_landmark)

                if not is_head_position_correct(angle_deg, selected_direction):
                    pamh.conversation(pamh.Message(pamh.PAM_TEXT_INFO, f"[INFO] ❌ Head position not valid for '{selected_direction}'."))
                    continue

                face_encoding = np.array(face_encoder.compute_face_descriptor(frame, face_landmark, 1))
                face_encoding /= np.linalg.norm(face_encoding)

                for model in target_models:
                    if "vault" in model:
                        result = unlock_vault(model["vault"], face_encoding.tolist())
                        if result:
                            return pamh.PAM_SUCCESS

        pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, "Face authentication failed."))
        return pamh.PAM_AUTH_ERR

    except Exception as e:
        pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, f"Error: {str(e)}"))
        return pamh.PAM_SYSTEM_ERR

def pam_sm_setcred(pamh, flags, argv):
    return pamh.PAM_SUCCESS

exported_functions = {
    "pam_sm_authenticate": pam_sm_authenticate,
    "pam_sm_setcred": pam_sm_setcred,
}
