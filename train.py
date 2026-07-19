# Import Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv("dataset/student.csv")

print("========== DATASET ==========")
print(data.head())

# -----------------------------
# Step 2: Check Dataset
# -----------------------------
print("\n========== DATASET INFO ==========")
print(data.info())

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# -----------------------------
# Step 3: Separate Features and Target
# -----------------------------
X = data.drop("Performance", axis=1)
y = data["Performance"]

# -----------------------------
# Step 4: Convert Output Labels into Numbers
# -----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nPerformance Classes:")
for i, label in enumerate(encoder.classes_):
    print(f"{i} --> {label}")

# -----------------------------
# Step 5: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))

# -----------------------------
# Step 6: Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Step 7: Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 8: Calculate Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print(f"Accuracy : {accuracy * 100:.2f}%")

# -----------------------------
# Step 9: Classification Report
# -----------------------------
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# -----------------------------
# Step 10: Save Model
# -----------------------------
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")

print("\nModel saved successfully in 'model/' folder.")
print("Training Completed Successfully!")