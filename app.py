from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

# Load trained ML model and TF-IDF vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Emotion emoji mapping
EMOTION_INFO = {
    "joy": ("😊", "JOY"),
    "happy": ("😊", "HAPPY"),
    "love": ("❤️", "LOVE"),
    "sadness": ("😢", "SADNESS"),
    "sad": ("😢", "SAD"),
    "anger": ("😡", "ANGER"),
    "angry": ("😡", "ANGRY"),
    "fear": ("😨", "FEAR"),
    "surprise": ("😲", "SURPRISE"),
    "disgust": ("🤢", "DISGUST"),
    "neutral": ("😐", "NEUTRAL")
}


HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>AI Text Emotion Detection</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #667eea,
                    #764ba2
                );

            display: flex;
            justify-content: center;
            align-items: center;

            padding: 25px;
        }

        .container {
            width: 100%;
            max-width: 700px;

            background: rgba(255,255,255,0.97);

            border-radius: 25px;

            padding: 35px;

            box-shadow:
                0 20px 50px rgba(0,0,0,0.25);
        }

        .title {
            text-align: center;

            font-size: 32px;
            font-weight: bold;

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
            background: #eeeeee;

            color: #444;
        }

        .result {
            margin-top: 30px;

            padding: 25px;

            border-radius: 20px;

            background: #f7f7ff;
        }

        .main-emotion {
            text-align: center;

            margin-bottom: 30px;
        }

        .main-emoji {
            font-size: 65px;
        }

        .main-name {
            font-size: 30px;

            font-weight: bold;

            color: #333;

            margin-top: 8px;
        }

        .confidence {
            margin-top: 8px;

            color: #666;

            font-size: 17px;
        }

        .probability-title {
            text-align: center;

            font-size: 21px;

            font-weight: bold;

            color: #444;

            margin-bottom: 25px;
        }

        .emotion-item {
            margin-bottom: 22px;
        }

        .emotion-name {
            display: flex;

            justify-content: space-between;

            align-items: center;

            font-size: 16px;

            font-weight: bold;

            color: #333;

            margin-bottom: 5px;
        }

        .percentage {
            color: #667eea;
        }

        .probability-line {
            font-family: monospace;

            font-size: 16px;

            letter-spacing: 1px;

            white-space: nowrap;

            overflow: hidden;

            color: #667eea;
        }

        .examples {
            text-align: center;

            margin-top: 25px;
        }

        .example {
            display: inline-block;

            background: #f0f0ff;

            padding: 9px 13px;

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

            .probability-line {
                font-size: 13px;
            }

        }

    </style>

</head>


<body>

<div class="container">

    <div class="title">
        🧠 AI Text Emotion Detection
    </div>

    <div class="subtitle">
        NLP Based Emotion Classification System
    </div>


    <form method="POST">

        <textarea
            name="text"
            id="text"
            placeholder="Enter your sentence here..."
            required>{{ text }}</textarea>


        <div class="buttons">

            <button
                type="submit"
                class="detect">

                🔍 Detect Emotion

            </button>


            <button
                type="button"
                class="clear"
                onclick="clearText()">

                🗑️ Clear

            </button>

        </div>

    </form>


    {% if emotion %}

    <div class="result">


        <!-- Main Prediction -->

        <div class="main-emotion">

            <div class="main-emoji">
                {{ emoji }}
            </div>

            <div class="main-name">
                {{ display_emotion }}
            </div>

            <div class="confidence">

                🎯 Confidence:
                <b>{{ confidence }}%</b>

            </div>

        </div>


        <!-- Probability Section -->

        <div class="probability-title">

            📊 Emotion Probabilities

        </div>


        {% for name, icon, probability in probabilities %}

        <div class="emotion-item">

            <div class="emotion-name">

                <span>
                    {{ icon }} {{ name }}
                </span>

                <span class="percentage">
                    {{ probability }}%
                </span>

            </div>


            <div class="probability-line">

                {% set filled =
                    ((probability / 100) * 20)
                    |round(0, 'floor')
                    |int
                %}

                {% set empty = 20 - filled %}

                {{ "━" * filled }}{{ "░" * empty }}

            </div>

        </div>

        {% endfor %}


    </div>

    {% endif %}


    <!-- Examples -->

    <div class="examples">

        <b>✨ Try Examples</b>

        <br>


        <span
            class="example"
            onclick="setText('I am very happy today!')">

            😊 Happy

        </span>


        <span
            class="example"
            onclick="setText('I feel very sad today')">

            😢 Sad

        </span>


        <span
            class="example"
            onclick="setText('This makes me so angry')">

            😡 Angry

        </span>


        <span
            class="example"
            onclick="setText('Wow! This is amazing!')">

            😲 Surprise

        </span>

    </div>


    <div class="footer">

        Powered by Python • Flask • NLP • TF-IDF • Machine Learning

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

        text = request.form.get(
            "text",
            ""
        ).strip()


        if text:

            # Convert text into TF-IDF features
            data = vectorizer.transform([text])


            # Predict emotion
            emotion = model.predict(data)[0]


            # Get prediction probabilities
            if hasattr(model, "predict_proba"):

                probs = model.predict_proba(data)[0]

                classes = model.classes_


                probability_data = []


                for cls, prob in zip(
                    classes,
                    probs
                ):

                    cls_key = str(cls).lower()

                    icon, name = EMOTION_INFO.get(
                        cls_key,
                        ("🤖", str(cls).upper())
                    )


                    probability_data.append(
                        (
                            name,
                            icon,
                            round(
                                float(prob) * 100,
                                1
                            )
                        )
                    )


                # Highest probability first
                probability_data.sort(
                    key=lambda x: x[2],
                    reverse=True
                )


                probabilities = probability_data


                confidence = round(
                    float(max(probs)) * 100,
                    1
                )


            # Main emotion display
            emoji, display_emotion = EMOTION_INFO.get(
                str(emotion).lower(),
                (
                    "🤖",
                    str(emotion).upper()
                )
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
