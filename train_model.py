import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("dataset/emotions.csv")

print("===================================")
print("   TEXT EMOTION DETECTION - NLP")
print("===================================")

print("\nDataset Loaded Successfully!")
print("Total Records:", len(df))

print("\nEmotion Counts:")
print(df["emotion"].value_counts())

# Input and output
X = df["text"]
y = df["emotion"]

# Convert text into numerical values
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train ML model
model = LogisticRegression(max_iter=1000)

model.fit(X_vectorized, y)

# Save trained model
joblib.dump(model, "emotion_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n===================================")
print("Model Training Completed!")
print("===================================")
print("Model saved as: emotion_model.pkl")
print("Vectorizer saved as: vectorizer.pkl")