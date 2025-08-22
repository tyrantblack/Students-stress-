import streamlit as st
import pandas as pd
import numpy as np
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

# Suggestions for each level
def get_suggestions(risk_level):
    if risk_level == "Low Risk":
        return [
            "👍 Keep maintaining a healthy balance between study and rest.",
            "💤 Ensure consistent sleep (7–8 hrs) to remain stress-free.",
            "🏃 Stay active with light physical exercise or walking.",
            "🤝 Continue engaging in social and family activities."
        ]
    elif risk_level == "Moderate Risk":
        return [
            "📅 Try making a daily study plan to avoid last-minute stress.",
            "🧘 Practice relaxation techniques like deep breathing or meditation.",
            "🍎 Maintain proper meals and avoid skipping food during busy days.",
            "📞 Talk to friends or mentors when feeling overwhelmed."
        ]
    else:  # High Risk
        return [
            "🚨 Prioritize mental health—consider reaching out to a counselor.",
            "💡 Break study sessions into smaller chunks with breaks.",
            "🛌 Improve sleep hygiene: fixed bedtime, no screens before sleep.",
            "⚡ Limit caffeine and energy drinks, replace with water or healthy alternatives.",
            "🏃 Engage in regular exercise to reduce stress hormones.",
            "🤝 Seek emotional support from family/friends without hesitation."
        ]

# ==================================================
# Streamlit UI
# ==================================================
st.title("🎓 Student Stress Prediction App")
st.write("Enter student details below and get stress prediction with risk classification & suggestions.")

# ------------------------------
# Student Details Section
# ------------------------------
st.header("📋 Student Details")

age = st.number_input("Age", min_value=18, max_value=30, value=21)
gender = st.selectbox("Gender", encoder_dict["Gender"].classes_)
acc_status = st.selectbox("Accommodation Status", encoder_dict["Accomodation status"].classes_)
sleep = st.number_input("Sleep Duration (hrs)", min_value=3, max_value=12, value=6)
screen = st.number_input("Screen Time (hrs)", min_value=0, max_value=12, value=8)
study = st.number_input("Study Hours", min_value=0, max_value=10, value=3)
meals = st.number_input("Meals per Day", min_value=1, max_value=5, value=3)
phy = st.selectbox("Physical Activity", encoder_dict["Physical activity"].classes_)
bev = st.selectbox("Beverage Intake", encoder_dict["Beverage intake"].classes_)

# ------------------------------
# Survey Questions Section
# ------------------------------
st.header("📝 Survey Questions (Yes = 1, No = 0)")

qs_labels = [
    "Q1. Do you feel academic pressure?",
    "Q2. Do you have difficulty concentrating?",
    "Q3. Do you experience frequent headaches?",
    "Q4. Do you feel anxious before exams?",
    "Q5. Do you have sleep disturbances?",
    "Q6. Do you feel supported by family/friends?",
    "Q7. Do you feel time management is difficult?",
    "Q8. Do you feel physically exhausted?",
    "Q9. Do you feel socially isolated?",
    "Q10. Do you face financial stress?"
]

qs = []
cols = st.columns(2)  # two columns for better readability
for i, q in enumerate(qs_labels):
    with cols[i % 2]:
        qs.append(st.selectbox(q, ["No", "Yes"], key=f"Q{i+1}"))

# Encode Yes/No to 1/0
qs_encoded = [1 if ans == "Yes" else 0 for ans in qs]

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
    **{f"Q{i+1}": qs_encoded[i] for i in range(10)}
}])

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("🔮 Predict Stress"):
    pred_score = model.predict(new_student)[0]
    pred_risk = classify_risk(pred_score)

    st.subheader("📊 Prediction Results")
    st.success(f"**Predicted Stress Score:** {pred_score:.2f}")
    st.info(f"**Risk Classification:** {pred_risk}")

    # Suggestions
    st.subheader("💡 Suggestions")
    suggestions = get_suggestions(pred_risk)
    for s in suggestions:
        st.write("- " + s)
