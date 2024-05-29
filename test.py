import speech_recognition as sr
from langdetect import detect
from googletrans import Translator
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play
import pyaudio

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Please speak into the microphone...")
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None

def detect_language(text):
    lang = detect(text)
    print(f"Detected language: {lang}")
    return lang

def translate_text(text, target_lang):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    print(f"Translated text: {translation.text}")
    return translation.text

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    
    # Convert mp3 to wav for playback
    sound = AudioSegment.from_mp3("output.mp3")
    sound.export("output.wav", format="wav")
    
    # Play the audio
    play(AudioSegment.from_wav("output.wav"))

if __name__ == "__main__":
    # Step 1: Recognize speech from microphone
    input_text = recognize_speech_from_mic()
    if input_text is None:
        exit()
    
    # Step 2: Detect language of the input text
    input_lang = detect_language(input_text)
    
    # Step 3: Ask user for target language
    target_lang = input("Enter the target language code (e.g., 'en' for English, 'es' for Spanish): ")
    
    # Step 4: Translate text to the target language
    translated_text = translate_text(input_text, target_lang)
    
    # Step 5: Convert translated text to speech and play
    text_to_speech(translated_text, target_lang)
    
    # Clean up temporary files
    os.remove("output.mp3")
    os.remove("output.wav")