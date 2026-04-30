import streamlit as st
import pandas as pd
import pickle
import os

st.title("Autism Prediction App")

# ---------------- LOAD MODEL SAFELY ---------------- #
model = None
encoders = None

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "best_model.pkl")
    encoder_path = os.path.join(BASE_DIR, "encoders.pkl")

    model = pickle.load(open(model_path, "rb"))
    encoders = pickle.load(open(encoder_path, "rb"))

    st.success("Model loaded successfully ✅")

except Exception as e:
    st.error("Model load failed ❌")
    st.write(e)

# ---------------- STOP IF MODEL NOT LOADED ---------------- #
if model is None or encoders is None:
    st.stop()

# ---------------- USER INPUT ---------------- #

st.header("Enter Details")

# A1–A10 (binary)
A1 = st.selectbox("A1 Score", [0,1])
A2 = st.selectbox("A2 Score", [0,1])
A3 = st.selectbox("A3 Score", [0,1])
A4 = st.selectbox("A4 Score", [0,1])
A5 = st.selectbox("A5 Score", [0,1])
A6 = st.selectbox("A6 Score", [0,1])
A7 = st.selectbox("A7 Score", [0,1])
A8 = st.selectbox("A8 Score", [0,1])
A9 = st.selectbox("A9 Score", [0,1])
A10 = st.selectbox("A10 Score", [0,1])

# Age (slider — better UI)
age = st.slider("Age", 1, 100, 18)

gender = st.selectbox("Gender", encoders["gender"].classes_)
ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_)
jaundice = st.selectbox("Jaundice", encoders["jaundice"].classes_)
austim = st.selectbox("Family ASD", encoders["austim"].classes_)
country = st.selectbox("Country", encoders["contry_of_res"].classes_)
used_app = st.selectbox("Used App Before", encoders["used_app_before"].classes_)
relation = st.selectbox("Relation", encoders["relation"].classes_)

# result score input
result = st.number_input("Result Score", min_value=0)

# ---------------- PREDICT ---------------- #

if st.button("Predict"):

    input_dict = {
        "A1_Score": A1,
        "A2_Score": A2,
        "A3_Score": A3,
        "A4_Score": A4,
        "A5_Score": A5,
        "A6_Score": A6,
        "A7_Score": A7,
        "A8_Score": A8,
        "A9_Score": A9,
        "A10_Score": A10,
        "age": age,
        "gender": encoders["gender"].transform([gender])[0],
        "ethnicity": encoders["ethnicity"].transform([ethnicity])[0],
        "jaundice": encoders["jaundice"].transform([jaundice])[0],
        "austim": encoders["austim"].transform([austim])[0],
        "contry_of_res": encoders["contry_of_res"].transform([country])[0],
        "used_app_before": encoders["used_app_before"].transform([used_app])[0],
        "relation": encoders["relation"].transform([relation])[0],
        "result": result
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("High chances of Autism ⚠️")
    else:
        st.success("Low chances of Autism ✅")