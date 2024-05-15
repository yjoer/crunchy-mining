import mlflow
import streamlit as st

st.set_page_config(layout="wide")
mlflow.set_tracking_uri("http://localhost:5001")
