import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import data1, data, data2
from main import shape1, shape2, shape3
from main import reports, fig1, fig2, custom_prediction

st.title("Advance SMS Spam Detection")

st.header("Datasets")

col1, col2, col3 = st.columns(3)
with col1:
    btn1= st.button("Dataset1")
with col2:
    btn2 = st.button("Dataset2")
with col3:
    btn3= st.button("Dataset3")

# Show dataset based on button clicked
if btn1:
    st.subheader("Dataset1: SMSSpamCollection")
    st.write(data1)
    st.write("Dataset1 shape:",shape1)
elif btn2:
    st.subheader("Dataset2: extra_spam.csv")
    st.write(data2)
    st.write("Dataset2 shape:", shape2)
elif btn3:
    st.subheader("Dataset3: combine Dataset")
    st.write(data)
    st.write("Dataset3:", shape3)

# Models classification report
# Create buttons in one line
st.subheader("Classification reports")

st.subheader("Naive Bayes")
st.text_area("Naive Bayes Report", reports["Naive Bayes"], height=200)

st.subheader("Logistic Regression")
st.text_area("Logistic Regression Report", reports["Logistic Regression"], height=200)

st.subheader("Linear SVM")
st.text_area("Linear SVM Report", reports["Linear SVM"], height=200)

st.subheader("Random Forest")
st.text_area("Random Forest Report", reports["Random Forest"], height=200)

st.subheader("XGBoost")
st.text_area("XGBoost Report", reports["XGBoost"], height=200)


#show charts 
st.subheader("Model accuracy comparision")
st.pyplot(fig1())

st.subheader("Model F1 score comparision")
st.pyplot(fig2())

# Show custome prediction
st.subheader("Custom prediction")
if st.button("Run Predictions"):
    results = custom_prediction()
    for r in results:
        st.write(f"**Message:** {r['Message']}")
        st.write(f"Prediction: {r['Prediction']}")
        st.write(f"Spam Probability: {r['Spam Probability']} | Ham Probability: {r['Ham Probability']}")
        st.write("---")