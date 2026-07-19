from flask import Flask, render_template, request
import joblib
import numpy as np

# Create Flask App
app = Flask(__name__)

# Load Trained Model and Encoder
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    previous_marks = float(request.form["previous_marks"])
    assignments = int(request.form["assignments"])
    sleep_hours = float(request.form["sleep_hours"])
    internet_usage = float(request.form["internet_usage"])

    student_data = np.array([[attendance,
                              study_hours,
                              previous_marks,
                              assignments,
                              sleep_hours,
                              internet_usage]])

    prediction = model.predict(student_data)
    result = encoder.inverse_transform(prediction)[0]

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)