import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ==================================================
# Load Dataset
# ==================================================
df = pd.read_excel("stress survey ml.xlsx", sheet_name="Sheet1")

# Drop unused
df = df.drop(columns=["Timestamp", "Stress_Level"], errors="ignore")

# Encode categorical
categorical_cols = ["Gender", "Accomodation status", "Physical activity", "Beverage intake"]
encoder_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoder_dict[col] = le

# Features & target
X = df.drop(columns=["Stress_Score"])
y = df["Stress_Score"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X_train, y_train)

# Risk classification
def classify_risk(score):
    if score <= 3:
        return "Low Risk"
    elif score <= 6:
        return "Moderate Risk"
    else:
        return "High Risk"

# ==================================================
# Streamlit UI
# ==================================================
st.title("🎓 Student Stress Prediction App")

st.sidebar.header("Enter Student Data")

# Input fields
age = st.sidebar.number_input("Age", 18, 30, 21)
gender = st.sidebar.selectbox("Gender", encoder_dict["Gender"].classes_)
acc_status = st.sidebar.selectbox("Accommodation Status", encoder_dict["Accomodation status"].classes_)
sleep = st.sidebar.slider("Sleep Duration (hrs)", 3, 12, 6)
screen = st.sidebar.slider("Screen Time (hrs)", 0, 12, 8)
study = st.sidebar.slider("Study Hours", 0, 10, 3)
meals = st.sidebar.slider("Meals per Day", 1, 5, 3)
phy = st.sidebar.selectbox("Physical Activity", encoder_dict["Physical activity"].classes_)
bev = st.sidebar.selectbox("Beverage Intake", encoder_dict["Beverage intake"].classes_)

st.sidebar.subheader("Survey Questions (Q1–Q10)")
qs = [st.sidebar.slider(f"Q{i}", 0, 1, 1) for i in range(1, 11)]

# Prepare input
new_student = pd.DataFrame([{
    "Age": age,
    "Gender": encoder_dict["Gender"].transform([gender])[0],
    "Accomodation status": encoder_dict["Accomodation status"].transform([acc_status])[0],
    "sleep duration": sleep,
    "Screen time": screen,
    "Study_hours": study,
    "meals_per_day": meals,
    "Physical activity": encoder_dict["Physical activity"].transform([phy])[0],
    "Beverage intake": encoder_dict["Beverage intake"].transform([bev])[0],
    **{f"Q{i+1}": qs[i] for i in range(10)}
}])

# Predict
if st.button("Predict Stress"):
    pred_score = model.predict(new_student)[0]
    pred_risk = classify_risk(pred_score)

    st.subheader("Prediction Results")
    st.write(f"**Predicted Stress Score:** {pred_score:.2f}")
    st.write(f"**Risk Classification:** {pred_risk}")

    # Feature importance
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(feature_names)), importances[indices], color="red")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    for bar, imp in zip(bars, importances[indices]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f"{imp:.3f}", ha='center', va='bottom', fontsize=8)
    st.pyplot(fig)
