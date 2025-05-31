from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Prompt the user to enter their journal thought
user_input = input("What's on your mind today? 💭\n> ")

# Define the possible categories to generalize into
categories = [
    "depression", 
    "anxiety", 
    "academic stress", 
    "imposter syndrome", 
    "loneliness", 
    "food insecurity", 
    "relationship issues", 
    "burnout",
    "family issues",
    "medical issues",
    "postive life",
    "good social life",
    "energetic",
    "humor",
    "success",
    "fear of death",
    "self-care",
    "feelings of inadeqaucy",
    "external-negative influence",
    "external-positive influence",
    "active-lifestyle",
]

# Run the classifier
result = classifier(user_input, categories)

# Show the top matching category
print("\n Most likely topic:", result['labels'][0])

# Optionally show all category scores
print("\n Confidence scores:")
for label, score in zip(result['labels'], result['scores']):
    print(f"• {label}: {score:.2f}")
