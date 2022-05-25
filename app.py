import numpy as np
from flask import Flask, request,render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [int(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction == "yes":
        return render_template("index.html", prediction_text = "Landslide may occur..!!!⚠")
    else:
        return render_template("index.html", prediction_text = "No chance for Landslide")

if __name__ == "__main__":
    flask_app.run(debug=True)