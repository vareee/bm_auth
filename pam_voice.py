import os
os.environ["PULSE_SERVER"] = "127.0.0.1"

import numpy as np
import librosa
import sounddevice as sd
from sklearn.metrics.pairwise import cosine_similarity
import random
import speech_recognition as sr


VOICE_SAMPLE_DIR = "/var/local/voice_samples"

def get_voice_sample(username):
    sample_file = os.path.join(VOICE_SAMPLE_DIR, f"{username}.npy")
    if os.path.exists(sample_file):
        return np.load(sample_file)
    return None

def capture_audio(duration=3, sample_rate=16000):
    print("Говорите...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def extract_mfcc(audio, sample_rate=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features = np.hstack([np.mean(mfcc.T, axis=0),
                          np.mean(delta_mfcc.T, axis=0),
                          np.mean(delta2_mfcc.T, axis=0)])
    return features

def compare_mfcc(mfcc1, mfcc2):
    mfcc1 = mfcc1.reshape(1, -1)
    mfcc2 = mfcc2.reshape(1, -1)
    similarity = cosine_similarity(mfcc1, mfcc2)[0][0]
    return similarity

def generate_random_word():
    words = [
        "дом", "кот", "мир", "лес", "вода",
        "свет", "путь", "ключ", "знак", "звук",
        "голос", "шум", "вход", "выход", "город"
    ]
    return random.choice(words)

def recognize_speech(audio, sample_rate=16000):
    recognizer = sr.Recognizer()
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_data = sr.AudioData(audio_int16.tobytes(), sample_rate, 2)
    try:
        text = recognizer.recognize_google(audio_data, language="ru-RU")
        print(f"Распознанное слово: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Не удалось распознать речь.")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")
    return None

def authenticate_user(username, pamh=None):
    stored_mfcc = get_voice_sample(username)
    if stored_mfcc is None:
        message = "Образец голоса не найден. Запись нового образца."
        if pamh:
            pamh.conversation(pamh.Message(pamh.PAM_TEXT_INFO, message))
        else:
            print(message)
        return False

    expected_word = generate_random_word()
    message = f"Произнесите слово: {expected_word}"
    if pamh:
        pamh.conversation(pamh.Message(pamh.PAM_TEXT_INFO, message))
    else:
        print(message)

    audio = capture_audio(duration=3)
    spoken_mfcc = extract_mfcc(audio)

    recognized_word = recognize_speech(audio)
    if recognized_word != expected_word:
        message = f"Произнесённое слово ('{recognized_word}') не совпадает с ожидаемым ('{expected_word}')."
        if pamh:
            pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, message))
        else:
            print(message)
        return False

    similarity = compare_mfcc(stored_mfcc, spoken_mfcc)
    message = f"Косинусное сходство: {similarity}"
    if pamh:
        pamh.conversation(pamh.Message(pamh.PAM_TEXT_INFO, message))
    else:
        print(message)

    threshold = 0.6
    if similarity > threshold:
        message = "Голосовая аутентификация успешна."
        if pamh:
            pamh.conversation(pamh.Message(pamh.PAM_TEXT_INFO, message))
        else:
            print(message)
        return True
    else:
        message = "Голосовая аутентификация не удалась."
        if pamh:
            pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, message))
        else:
            print(message)
        return False

def pam_sm_authenticate(pamh, flags, argv):
    try:
        username = pamh.get_user(None)
        if not username:
            pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, "Не удалось определить имя пользователя."))
            return pamh.PAM_AUTH_ERR

        if authenticate_user(username, pamh=pamh):
            return pamh.PAM_SUCCESS
        else:
            return pamh.PAM_AUTH_ERR
    except Exception as e:
        pamh.conversation(pamh.Message(pamh.PAM_ERROR_MSG, f"Ошибка при аутентификации: {e}"))
        return pamh.PAM_AUTH_ERR

def pam_sm_setcred(pamh, flags, argv):
    return pamh.PAM_SUCCESS

exported_functions = {
    "pam_sm_authenticate": pam_sm_authenticate,
    "pam_sm_setcred": pam_sm_setcred,
}
