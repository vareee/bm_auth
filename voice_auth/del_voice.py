import os

VOICE_SAMPLE_DIR = "/var/local/voice_samples"

def delete_voice_sample(username):
    sample_file = os.path.join(VOICE_SAMPLE_DIR, f"{username}.npy")
    if os.path.exists(sample_file):
        os.remove(sample_file)
        print(f"Образец голоса для пользователя '{username}' удалён.")
    else:
        print(f"Образец голоса для пользователя '{username}' не найден.")

def main():
    username = os.getenv("USER")
    if not username:
        print("Не удалось определить имя пользователя.")
        exit(1)

    print(f"Текущий пользователь: {username}")
    confirm = input(f"Вы уверены, что хотите удалить голосовой образец для '{username}'? (y/n): ").strip().lower()

    if confirm == 'y':
        delete_voice_sample(username)
    else:
        print("Удаление отменено.")

if __name__ == "__main__":
    main()
