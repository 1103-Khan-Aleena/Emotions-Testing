from transformers import pipeline
from collections import defaultdict
#does measre neutral
MAX_ROWS = 100
#less accurate
# Initialize the Hugging Face pipeline for emotion classification
pipe = pipeline("text-classification", model="michellejieli/emotion_text_classifier")

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
            try:
                emotion = pipe(text)[0]['label'].lower()
                emotion_counts[emotion] += 1
                total_predictions += 1
            except Exception as e:
                print(f"Error analyzing line {i+1}: {e}")
        else:
            print(f"Skipping line {i+1}: bad format")

# Print final report
print("\n🎯 Emotion Distribution:")
for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
    percent = (count / total_predictions) * 100
    print(f"{emotion}: {count} ({percent:.2f}%)")

print(f"\n✅ Total Analyzed Lines: {total_predictions}")
print(f"🚫 Skipped Lines (e.g. 'neutral' or bad format): {skipped_lines}")
