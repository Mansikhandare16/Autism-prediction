import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Autism Prediction App")

@st.cache_resource
def train_model():
    df = pd.read_csv("train.csv")

    df = df.drop(columns=["ID", "age_desc"])

    df = pd.get_dummies(df)

    X = df.drop("Class/ASD", axis=1)
    y = df["Class/ASD"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, X.columns

model, feature_columns = train_model()

st.header("Enter Details")

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

age = st.slider("Age", 1, 100, 18)

result = st.number_input("Result Score", min_value=0)

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
        "result": result
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("High chances of Autism ⚠️")
    else:
        st.success("Low chances of Autism ✅")