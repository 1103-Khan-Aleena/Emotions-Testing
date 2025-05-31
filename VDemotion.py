import csv
import requests
from collections import defaultdict
#reads csv file (cant filter out the neutral emtions so we should kill it)
MAX_ROWS = 100

API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
headers = {
    "Authorization": "Bearer {API TOKEN}"
}

# Function to get emotion from Hugging Face
def detect_emotion(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        result = response.json()
        top_emotion = sorted(result[0], key=lambda x: x['score'], reverse=True)[0]['label']
        return top_emotion
    except Exception as e:
        print(f"Error: {e}")
        return "unknown"

# Emotion tracking
emotion_counts = defaultdict(int)
total_predictions = 0

print("📊 Analyzing...")

csv_file = 'tweet_emotions.csv'

with open(csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header

    for i, row in enumerate(reader):
        if i >= MAX_ROWS:
            break
        if len(row) >= 3:
            text = row[2]
            emotion = detect_emotion(text)
            emotion_counts[emotion] += 1
            total_predictions += 1
        else:
            print(f"Skipping row {i+1}: not enough columns")

# Print final report
print("\n Emotion Distribution:")
for emotion, count in emotion_counts.items():
    percent = (count / total_predictions) * 100
    print(f"{emotion}: {count} ({percent:.2f}%)")

print(f"\n Total Analyzed Tweets: {total_predictions}")
