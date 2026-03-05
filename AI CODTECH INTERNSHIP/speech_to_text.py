import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Load audio file
audio_file = "audio.wav"

try:
    with sr.AudioFile(audio_file) as source:
        print("Reading audio file...")
        audio_data = recognizer.record(source)

    print("Transcribing audio...")

    # Convert speech to text
    text = recognizer.recognize_google(audio_data)

    print("\n===== TRANSCRIPTION RESULT =====\n")
    print(text)

except FileNotFoundError:
    print("Audio file not found. Please place audio.wav in this folder.")

except sr.UnknownValueError:
    print("Speech could not be understood.")

except sr.RequestError:
    print("Could not connect to the speech recognition service.")