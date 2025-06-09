import os
import numpy as np
import librosa
import sounddevice as sd
from sklearn.metrics.pairwise import cosine_similarity


VOICE_SAMPLE_DIR = "/var/local/voice_samples"
os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)

def save_voice_sample(username, mfcc):
    sample_file = os.path.join(VOICE_SAMPLE_DIR, f"{username}.npy")
    np.save(sample_file, mfcc)
    print(f"Образец голоса для пользователя '{username}' сохранён.")

def capture_audio(duration=7, sample_rate=16000):
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

def record_reference_sample(num_phrases=10, duration=7, sample_rate=16000):
    print("Запись референсного образца голоса.")
    print("Просто говорите что-то в течение следующих нескольких записей...")
    all_features = []

    for i in range(num_phrases):
        print(f"Запись {i + 1}/{num_phrases}... Говорите что-нибудь!")
        audio = capture_audio(duration, sample_rate)
        mfcc = extract_mfcc(audio, sample_rate)
        all_features.append(mfcc)

    reference_mfcc = np.mean(all_features, axis=0)
    return reference_mfcc

def main():
    username = os.getenv("USER")
    if not username:
        print("Не удалось определить имя пользователя.")
        exit(1)

    print(f"Текущий пользователь: {username}")

    print("Режим записи референсного образца голоса.")
    reference_mfcc = record_reference_sample()
    save_voice_sample(username, reference_mfcc)

if __name__ == "__main__":
    main()
