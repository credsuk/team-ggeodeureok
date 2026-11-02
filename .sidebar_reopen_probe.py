import streamlit as st
from src.app import APP_STYLE

st.set_page_config(page_title="Sidebar reopen verification", layout="wide", menu_items={})
st.markdown(APP_STYLE, unsafe_allow_html=True)
st.title("Sidebar reopen verification")
with st.sidebar:
    st.write("Sidebar content")
st.write("Main content")
