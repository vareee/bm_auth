import numpy as np
import sounddevice as sd
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture
import argparse
import sys
import random
import speech_recognition as sr
import librosa


sample_rate = 16000
duration = 3
n_components = 8
challenge_words = ["семь", "дом", "путь", "роль", "час", "шар", "лед", "тип"]
n_train_phrases = 5

def record_audio():
    print("Запись началась... Говорите сейчас.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def extract_features(audio):
    mfcc_feat = mfcc(audio, sample_rate, numcep=13, 
                    winlen=0.025, winstep=0.01,
                    nfft=2048, preemph=0.97, appendEnergy=True)
    delta = librosa.feature.delta(mfcc_feat)
    delta_delta = librosa.feature.delta(mfcc_feat, order=2)
    return np.hstack((mfcc_feat, delta, delta_delta))

def train_gmm(features_list):
    gmm = GaussianMixture(n_components=n_components, 
                         covariance_type='diag',
                         max_iter=200,
                         n_init=3)
    gmm.fit(np.vstack(features_list))
    return gmm

def save_reference_voice():
    print(f"Запишите {n_train_phrases} разных фраз для обучения:")
    features = []
    for i in range(n_train_phrases):
        print(f"Фраза {i+1}: Произнесите любую случайную фразу")
        audio = record_audio()
        features.append(extract_features(audio))
    
    model = train_gmm(features)
    np.save("gmm_model.npy", model.weights_)
    print("Модель голоса сохранена")

def recognize_speech(audio):
    r = sr.Recognizer()
    audio_data = sr.AudioData(
        (audio * 32767).astype(np.int16).tobytes(),
        sample_rate,
        2
    )
    try:
        return r.recognize_google(audio_data, language="ru-RU").lower()
    except:
        return None

def authenticate(challenge_word):
    audio = record_audio()
    
    recognized_text = recognize_speech(audio)
    print(f"Распознано: {recognized_text}")
    
    if recognized_text != challenge_word:
        print("Ошибка: Неверное слово!")
        return False
    
    try:
        model_weights = np.load("gmm_model.npy")
    except FileNotFoundError:
        print("Ошибка: Модель не найдена!")
        sys.exit(1)
 
    features = extract_features(audio)
    gmm = GaussianMixture(n_components=n_components, 
                         weights_init=model_weights,
                         covariance_type='diag')
    gmm.fit(features)
    
    log_prob = gmm.score(features)
    print(f"Логарифмическая вероятность: {log_prob:.2f}")
    
    threshold = -50
    return log_prob > threshold

def main():
    parser = argparse.ArgumentParser(description="Голосовая аутентификация")
    parser.add_argument("--train", action="store_true", help="Обучить GMM модель")
    args = parser.parse_args()

    if args.train:
        save_reference_voice()
    else:
        challenge = random.choice(challenge_words)
        print(f"Пожалуйста, произнесите слово: {challenge}")
        
        if authenticate(challenge):
            print("Аутентификация успешна!")
            sys.exit(0)
        else:
            print("Доступ запрещен!")
            sys.exit(1)

if __name__ == "__main__":
    main()
