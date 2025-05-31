import requests
from collections import defaultdict
#Most accurate so far does not meaure neutral measures sad, joy, anger, love and fear
MAX_ROWS = 100

API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
headers = {
    "Authorization": "Bearer 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"
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
skipped_lines = 0

print(" Analyzing...")

txt_file = 'val.txt'

with open(txt_file, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i >= MAX_ROWS:
            break
        parts = line.strip().split(';')
        if len(parts) == 2:
            text, label = parts[0].strip(), parts[1].strip().lower()
            if label == "neutral":
                skipped_lines += 1
                continue
            emotion = detect_emotion(text)
            emotion_counts[emotion] += 1
            total_predictions += 1
        else:
            print(f"Skipping line {i+1}: bad format")

# Print final report
print("\n Emotion Distribution:")
for emotion, count in emotion_counts.items():
    percent = (count / total_predictions) * 100
    print(f"{emotion}: {count} ({percent:.2f}%)")

print(f"\n Total Analyzed Lines: {total_predictions}")
print(f" Skipped Lines (e.g. neutral or bad format): {skipped_lines}")
