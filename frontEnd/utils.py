import streamlit as st
import time
import requests
import json

BACKEND_URL = 'http://localhost:5000'

sentimentMap = {
    0 : "Irrelevant", 1 : "Negative", 2 : "Neutral", 3 : "Positive"
}

sentimentColorMap = {
    0 : "red", 1 : "orange", 2 : "gray", 3 : "blue"
}

def login():
    with st.form("login"):
        st.write("### Connexion")
        email = st.text_input("###### Your email",placeholder="dubois@gmail.com")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
                response = requests.post(
                    BACKEND_URL+"/login",
                    json={
                        "password" : password
                    }
                )
                if response.status_code == 200:
                    st.session_state["email"] = email
                    time.sleep(2)
                    st.success("Logged successfully")
                    st.rerun() 

                else:
                    st.error("Password is not correct")



def sentimentAnalysis():
    
    st.write("### Sentiment analysis")
    text = st.text_area("Text to process")
    submitted = st.button("Validate", type="primary")
    
    st.markdown("***")
    if submitted:
        response = requests.post(
            BACKEND_URL+"/predict",
            json={
                "text" : text
            }
        )

        sentiment = json.loads(response.text)["sentiment"]

        st.write("#### Result : :{}[{}]".format(sentimentColorMap[sentiment], sentimentMap[sentiment]))
    