import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import smtplib
import cv2
import requests
import numpy as np
from ultralytics import YOLO
import pytesseract
from PIL import Image

config = {
    'email_to': "abdulhady961@gmail.com",
    'browser': "brave",
    'gemini_api_key': 'AIzaSyD9y8ik6grTgeFDx28QHX9puH-Q70AflG0',  
    'email_from': "your_email@gmail.com",
    'email_password': "your_password",
    'yolo_model_path': "C:/Users/abdul/Desktop/BME/yolov8x.pt"
}

engine = pyttsx3.init('sapi5')
engine.setProperty('voice', engine.getProperty('voices')[0].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def wishMe():
    hour = datetime.datetime.now().hour
    if hour < 12:
        speak("Good Morning! Hope you're having a great day!")
    elif 12 <= hour < 18:
        speak("Good Afternoon! Ready to tackle the rest of the day?")
    else:
        speak("Good Evening! Relax, I'm here to help.")
    speak("Hello! I'm BME, your assistant. What would you like to do today?")

def takeCommand():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        speak("I'm all ears. Please tell me what you'd like to do.")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio, language='en-in')
        print(f"User said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that. Can you repeat?")
    except sr.RequestError:
        speak("Sorry, there seems to be an issue with the service.")
    return None

def identify_objects():
    speak("Initializing object detection. Please wait.")
    model = YOLO(config['yolo_model_path'])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            speak("I couldn't get a clear frame. Please check your camera.")
            break

        results = model(frame)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = box.conf[0].item()
                label = model.names[int(box.cls[0].item())]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detection_text = f"{label} detected with confidence {confidence:.2f}"
                print(detection_text)
                speak(detection_text)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Stopping object detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

def live_ocr_reading():
    speak("Starting live OCR reading. Point your camera at text. Press 'q' to stop.")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_text = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            speak("I couldn't get a clear frame. Please check your camera.")
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray, lang='eng')

        text = text.strip()
        if text and text != last_text:
            print(f"Detected text: {text}")
            speak(f"Reading: {text}")
            last_text = text

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Live OCR Reading', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Stopping OCR reading.")
            break

    cap.release()
    cv2.destroyAllWindows()

def search_gemini(query):
    try:
        genai.configure(api_key=config['gemini_api_key'])
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        speak("I couldn't process your search right now. Please try again later.")
        return None

def handle_search():
    speak("What would you like to search?")
    query = takeCommand()
    if query:
        response = search_gemini(query)
        if response:
            print(response)
            speak(response)

if __name__ == "__main__":
    wishMe()

    while True:
        query = takeCommand()
        if query:
            print(f"Received command: {query}")
            if "exit" in query:
                speak("Goodbye! Have a great day!")
                break
            elif "object" in query:
                identify_objects()
            elif "search" in query:
                handle_search()
            elif "ocr" in query or "read text" in query:
                live_ocr_reading()
            else:
                speak("I didn't understand that. Please try again.")