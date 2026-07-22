"""Streamlit interface: paste an email, get a phishing/legitimate verdict."""

import streamlit as st

st.title("Phishing Email Detector")

email_text = st.text_area("Paste an email to analyze", height=250)

if st.button("Check email"):
    if not email_text.strip():
        st.warning("Please paste some email text first.")
    else:
        # TODO: replace with real model call once training pipeline is ready
        st.info("Model not yet connected — this is a placeholder.")
