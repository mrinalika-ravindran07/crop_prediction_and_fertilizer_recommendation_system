import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ---------------- Load Dataset ----------------
df = pd.read_csv("INDIANCROP_DATA_cleaned.csv")
df = df.drop_duplicates()

target = df['CROP']
features = df[['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']]

# Train-test split (70:30)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.3, random_state=42)

# Train the model once
RF = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=7)
RF.fit(Xtrain, Ytrain)

# ---------------- Streamlit UI ----------------
st.title("🌾 Crop Recommendation and Fertilizer Suggestion System")
st.write("Predict the most suitable crops based on soil and weather conditions.")

# Sidebar input section
st.sidebar.header("Enter Soil and Weather Values")
N_SOIL = st.sidebar.number_input("Nitrogen (N_SOIL %):", min_value=0, max_value=200, value=50)
P_SOIL = st.sidebar.number_input("Phosphorus (P_SOIL %):", min_value=0, max_value=200, value=50)
K_SOIL = st.sidebar.number_input("Potassium (K_SOIL %):", min_value=0, max_value=200, value=50)
TEMPERATURE = st.sidebar.number_input("Temperature (°C):", min_value=0.0, max_value=60.0, value=25.0)
HUMIDITY = st.sidebar.number_input("Humidity (%):", min_value=0.0, max_value=100.0, value=60.0)
ph = st.sidebar.number_input("pH value:", min_value=0.0, max_value=14.0, value=6.5)
RAINFALL = st.sidebar.number_input("Rainfall (mm):", min_value=0.0, max_value=500.0, value=100.0)

# Fertilizer suggestion function
def recommend_fertilizer(N, P, K):
    recs = []
    if N < 40:
        recs.append("Nitrogen low → Use Urea / Ammonium Sulphate")
    elif N > 80:
        recs.append("Nitrogen high → Reduce Urea, use organic manure")
    if P < 40:
        recs.append("Phosphorus low → Use Single Super Phosphate (SSP)")
    elif P > 80:
        recs.append("Phosphorus high → Avoid DAP, balance with compost")
    if K < 40:
        recs.append("Potassium low → Use Muriate of Potash (MOP)")
    elif K > 80:
        recs.append("Potassium high → Avoid excess MOP, apply bio-fertilizers")
    if not recs:
        recs.append("Soil nutrients are balanced ✅")
    return recs

# ------------- Predict only when button is clicked -------------
if st.sidebar.button("Predict Crop"):
    data = np.array([[N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL]])
    proba = RF.predict_proba(data)[0]
    top3_idx = np.argsort(proba)[-3:][::-1]
    top3_crops = [RF.classes_[i] for i in top3_idx]
    top3_probs = [proba[i] * 100 for i in top3_idx]

    st.subheader("🌾 Top 3 Recommended Crops")
    for crop, prob in zip(top3_crops, top3_probs):
        st.write(f"- **{crop}** — {prob:.2f}% suitability")

    st.subheader("💡 Fertilizer Suggestions")
    for rec in recommend_fertilizer(N_SOIL, P_SOIL, K_SOIL):
        st.write("- " + rec)

else:
    st.info("👈 Enter your soil and weather details, then click **Predict Crop** to see results.")

# ---------------- Model performance ----------------
st.write("---")
st.subheader("📊 Model Performance (on Test Data)")
train_acc = RF.score(Xtrain, Ytrain) * 100
test_acc = RF.score(Xtest, Ytest) * 100
st.write(f"**Training Accuracy:** {train_acc:.2f}%")
st.write(f"**Testing Accuracy:** {test_acc:.2f}%")

# Confusion matrix
predicted_values = RF.predict(Xtest)
conf_matrix = confusion_matrix(Ytest, predicted_values)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
st.pyplot(fig)
