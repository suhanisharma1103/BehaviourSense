import streamlit as st
import requests

st.title("BehaviorSense Risk Detection")

st.write("Analyze messages for phishing, stress, and toxicity.")

user_input = st.text_area("Enter a message")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:

        url = "http://127.0.0.1:8000/analyze"   # your FastAPI endpoint

        response = requests.post(
            url,
            json={"text": user_input}
        )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Risk Scores")

            st.write("Phishing Score:", data["phishing_score"])
            st.write("Stress Score:", data["stress_score"])
            st.write("Toxicity Score:", data["toxicity_score"])
            st.write("Final Risk:", data["final_risk"])

        else:
            st.error("API error")