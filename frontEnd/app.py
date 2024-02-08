import streamlit as st
import time
from utils import login, sentimentAnalysis


if "email" not in st.session_state:
    login()
else:
    sentimentAnalysis()