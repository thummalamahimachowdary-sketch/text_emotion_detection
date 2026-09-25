from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

EMOTION_INFO = {
    "joy": ("😊", "Joy"),
    "happy": ("😄", "Happy"),
    "sadness": ("😢", "Sadness"),
    "sad": ("😢", "Sad"),
    "anger": ("😡", "Anger"),
    "angry": ("😡", "Angry"),
    "fear": ("😨", "Fear"),
    "surprise": ("😲", "Surprise"),
    "disgust": ("🤢", "Disgust"),
    "love": ("❤️", "Love"),
    "neutral": ("😐", "Neutral")
}

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Text Emotion Detection</title>

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 25px;
        }

        .container {
            width: 100%;
            max-width: 650px;
            background: rgba(255,255,255,0.97);
            border-radius: 25px;
            padding: 35px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.25);
        }

        .title {
            text-align: center;
            font-size: 32px;
            color: #333;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            height: 120px;
            padding: 18px;
            border: 2px solid #ddd;
            border-radius: 15px;
            resize: none;
            font-size: 17px;
            outline: none;
        }

        textarea:focus {
            border-color: #667eea;
        }

        .buttons {
            display: flex;
            gap: 12px;
            margin-top: 15px;
        }

        button {
            flex: 1;
            padding: 14px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        .detect {
            background: #667eea;
            color: white;
        }

        .clear {
            background: #eee;
            color: #444;
        }

        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 20px;
            background: #f7f7ff;
            text-align: center;
        }

        .emoji {
            font-size: 65px;
        }

        .emotion {
            font-size: 30px;
            font-weight: bold;
            color: #333;
            margin: 10px;
        }

        .confidence {
            color: #666;
            font-size: 17px;
        }

        .bar-container {
            margin-top: 20px;
            text-align: left;
        }

        .bar-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 14px;
        }

        .bar {
            height: 10px;
            background: #e5e5e5;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 12px;
        }

        .fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
        }

        .examples {
            margin-top: 25px;
            text-align: center;
        }

        .example {
            display: inline-block;
            background: #f0f0ff;
            padding: 8px 12px;
            margin: 5px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
        }

        .footer {
            text-align: center;
            margin-top: 25px;
            color: #888;
            font-size: 13px;
        }

        @media(max-width:600px) {
            .container {
                padding: 22px;
            }

            .title {
                font-size: 25px;
            }

            .buttons {
                flex-direction: column;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="title">🧠 AI Text Emotion Detection</div>

    <div class="subtitle">
        NLP-based Emotion Classification System
    </div>

    <form method="POST">

        <textarea
            id="text"
            name="text"
            placeholder="Type your sentence here..."
            required>{{ text or "" }}</textarea>

        <div class="buttons">
            <button class="detect" type="submit">
                🔍 Detect Emotion
            </button>

            <button class="clear" type="button"
                    onclick="clearText()">
                🗑️ Clear
            </button>
        </div>

    </form>

    {% if emotion %}

    <div class="result">

        <div class="emoji">{{ emoji }}</div>

        <div class="emotion">
            {{ display_emotion }}
        </div>

        <div class="confidence">
            Confidence: <b>{{ confidence }}%</b>
        </div>

        <div class="bar-container">

            {% for name, probability in probabilities %}

            <div class="bar-label">
                <span>{{ name }}</span>
                <span>{{ probability }}%</span>
            </div>

            <div class="bar">
                <div class="fill"
                     style="width: {{ probability }}%">
                </div>
            </div>

            {% endfor %}

        </div>

    </div>

    {% endif %}

    <div class="examples">

        <b>Try examples</b><br>

        <span class="example"
              onclick="setText('I am very happy today!')">
            😊 Happy
        </span>

        <span class="example"
              onclick="setText('I feel very sad today')">
            😢 Sad
        </span>

        <span class="example"
              onclick="setText('This makes me so angry')">
            😡 Angry
        </span>

        <span class="example"
              onclick="setText('Wow! This is amazing!')">
            😲 Surprise
        </span>

    </div>

    <div class="footer">
        Powered by Python • NLP • TF-IDF • Machine Learning
    </div>

</div>

<script>

function setText(value) {
    document.getElementById("text").value = value;
}

function clearText() {
    document.getElementById("text").value = "";
}

</script>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    emotion = None
    display_emotion = None
    emoji = "🤖"
    confidence = 0
    probabilities = []
    text = ""

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if text:

            data = vectorizer.transform([text])

            emotion = model.predict(data)[0]

            # Prediction probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(data)[0]
                classes = model.classes_

                probability_data = []

                for cls, prob in zip(classes, probs):
                    probability_data.append(
                        (str(cls).title(), round(prob * 100, 1))
                    )

                probability_data.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                probabilities = probability_data

                confidence = round(max(probs) * 100, 1)

            emoji, display_emotion = EMOTION_INFO.get(
                str(emotion).lower(),
                ("🤖", str(emotion).title())
            )

    return render_template_string(
        HTML,
        emotion=emotion,
        display_emotion=display_emotion,
        emoji=emoji,
        confidence=confidence,
        probabilities=probabilities,
        text=text
    )


if __name__ == "__main__":
    app.run()
