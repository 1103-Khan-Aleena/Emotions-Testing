import csv
import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get token from environment
API_TOKEN = os.getenv("HF_TOKEN")

# Dataset splits
splits = {
    'train': 'split/train-00000-of-00001.parquet',
    'validation': 'split/validation-00000-of-00001.parquet',
    'test': 'split/test-00000-of-00001.parquet'
}

# Load parquet file from Hugging Face Hub
df = pd.read_parquet("hf://datasets/dair-ai/emotion/" + splits["train"])

# CSV file path
csv_file = 'twitter_validation.csv'

# Hugging Face API endpoint (note: this is dataset URL, not model inference API!)
API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Function to detect emotion
def detect_emotion(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        emotion = response.json()[0][0]["label"]
        return emotion.lower()
    except Exception as e:
        print(f"Error with Hugging Face API: {e}")
        return "unknown"

# Process the CSV file
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # skip header

    for row in csvreader:
        text = row[3]
        emotion = detect_emotion(text)
        print(f"Text: {text}")
        print(f"Emotion: {emotion}")
