import pandas as pd
import streamlit as st

from crunchy_mining.vis import plot_loan_default_rate_by_state
from crunchy_mining.vis import plot_top_state_by_loan

st.set_page_config(layout="wide")

df = pd.read_csv("data/SBA.csv")

cols = st.columns([1, 1])
top_state_by_loans = plot_top_state_by_loan(df)
cols[0].altair_chart(top_state_by_loans[0])
cols[1].altair_chart(top_state_by_loans[1])

cols = st.columns([1, 1])
cols[0].altair_chart(plot_loan_default_rate_by_state(df))
