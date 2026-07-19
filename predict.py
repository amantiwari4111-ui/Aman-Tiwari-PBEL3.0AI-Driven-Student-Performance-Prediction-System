import joblib
import numpy as np

# Load the trained model and label encoder
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")

print("========================================")
print(" STUDENT PERFORMANCE PREDICTION SYSTEM ")
print("========================================")

# Take input from the user
attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours per day: "))
previous_marks = float(input("Enter Previous Marks: "))
assignments = int(input("Assignments Submitted? (1 = Yes, 0 = No): "))
sleep_hours = float(input("Enter Sleep Hours per day: "))
internet_usage = float(input("Enter Internet Usage (hours/day): "))

# Store input in a NumPy array
student_data = np.array([[attendance,
                          study_hours,
                          previous_marks,
                          assignments,
                          sleep_hours,
                          internet_usage]])

# Predict
prediction = model.predict(student_data)

# Convert numeric prediction back to text
result = encoder.inverse_transform(prediction)

print("\n========================================")
print("Predicted Student Performance:", result[0])
print("========================================")