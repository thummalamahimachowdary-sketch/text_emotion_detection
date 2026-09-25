from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Text Emotion Detection</title>
    <style>
        body {
            font-family: Arial;
            text-align: center;
            margin-top: 80px;
            background: #f2f2f2;
        }
        .box {
            background: white;
            width: 500px;
            margin: auto;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 0 15px #aaa;
        }
        input {
            width: 80%;
            padding: 12px;
            font-size: 16px;
        }
        button {
            padding: 12px 25px;
            margin-top: 15px;
            font-size: 16px;
            cursor: pointer;
        }
        h2 {
            color: #444;
        }
    </style>
</head>

<body>

<div class="box">
    <h1>😊 Text Emotion Detection</h1>

    <form method="POST">
        <input type="text" name="text"
               placeholder="Enter your sentence..." required>

        <br>

        <button type="submit">Detect Emotion</button>
    </form>

    {% if emotion %}
        <h2>Detected Emotion: {{ emotion }}</h2>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    emotion = None

    if request.method == "POST":
        text = request.form["text"]

        data = vectorizer.transform([text])
        emotion = model.predict(data)[0]

    return render_template_string(HTML, emotion=emotion)


if __name__ == "__main__":
    app.run()