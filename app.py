import streamlit as st
import pickle
import numpy as np
import os

# ---------------- LOAD FILES ----------------

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

st.set_page_config(page_title="Autism Screening Tool", layout="centered")

# ---------------- SIDEBAR ----------------
st.sidebar.title("About")
st.sidebar.info("""
This app uses Machine Learning for early autism screening.

⚠️ Not a medical diagnosis.
Consult a professional for confirmation.
""")

# ---------------- HEADER ----------------
st.title("Autism Early Screening Tool")

st.markdown("""
This tool helps identify potential autism traits based on behavioral patterns.  
Please answer honestly for better results.
""")

progress = st.progress(0)

st.write("---")

# ---------------- PERSONAL INFO ----------------
with st.expander("Personal Information", expanded=True):

    age = st.selectbox("Select Age", list(range(1,101)))

    gender = st.selectbox("Gender", ["m", "f"])
    ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_)
    jaundice = st.selectbox("Born with jaundice?", ["yes", "no"])
    austim = st.selectbox("Family member with autism?", ["yes", "no"])
    country = st.selectbox("Country", encoders["contry_of_res"].classes_)
    used_app = st.selectbox("Used app before?", ["yes", "no"])
    relation = st.selectbox("Relation", encoders["relation"].classes_)

progress.progress(30)

st.write("---")

# ---------------- QUESTIONS ----------------
with st.expander("Behavioral Questions", expanded=True):

    def yes_no(q):
        return 1 if st.radio(q, ["Yes", "No"], horizontal=True) == "Yes" else 0

    col1, col2 = st.columns(2)

    with col1:
        a1 = yes_no("Enjoys social interaction?")
        a3 = yes_no("Responds to name?")
        a5 = yes_no("Maintains eye contact?")
        a7 = yes_no("Interested in other children?")
        a9 = yes_no("Understands instructions?")

    with col2:
        a2 = yes_no("Prefers to be alone?")
        a4 = yes_no("Avoids eye contact?")
        a6 = yes_no("Upset by small changes?")
        a8 = yes_no("Repeats actions?")
        a10 = yes_no("Communication difficulty?")

progress.progress(70)

st.write("---")

# ---------------- ENCODING ----------------
gender = encoders["gender"].transform([gender])[0]
ethnicity = encoders["ethnicity"].transform([ethnicity])[0]
jaundice = encoders["jaundice"].transform([jaundice])[0]
austim = encoders["austim"].transform([austim])[0]
country = encoders["contry_of_res"].transform([country])[0]
used_app = encoders["used_app_before"].transform([used_app])[0]
relation = encoders["relation"].transform([relation])[0]

# ---------------- CALCULATE RESULT ----------------
result = a1+a2+a3+a4+a5+a6+a7+a8+a9+a10

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    try:
        input_data = np.array([[
            a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,
            age,
            gender,
            ethnicity,
            jaundice,
            austim,
            country,
            used_app,
            result,
            relation
        ]])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        progress.progress(100)

        st.markdown("## Result Summary")

        st.write(f"Total Score: {result}/10")

        if prob < 0.3:
            st.success(f"Low Risk\n\nProbability: {prob*100:.2f}%")
        elif prob < 0.7:
            st.warning(f"Medium Risk\n\nProbability: {prob*100:.2f}%")
        else:
            st.error(f"High Risk\n\nProbability: {prob*100:.2f}%")

        # ---------------- RECOMMENDATIONS ----------------
        if prob > 0.7:
            st.markdown("""
### Recommendation:
- Consider consulting a specialist  
- Early intervention can be very helpful  
- Monitor behavioral patterns closely  
""")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------- FOOTER ----------------
st.write("---")
st.info("⚠️ This is not a medical diagnosis. Please consult a healthcare professional.")