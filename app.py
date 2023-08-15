import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

st.set_page_config(
    page_title="Detection Demo Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
#        'Get Help': 'https://www.extremelycoolapp.com/help',
#        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This page is developed by HVN"
    }
)

st.title(body='Detection Demo Dashboard', help='')
st.header('Metal surface detection')
#st.subheader('This is a subheader')
#st.caption('This is a string that explains something above.')

if 'run_detect' not in st.session_state:
    st.session_state['run_detect'] = False

def run_detection():
    st.session_state['run_detect'] = True

st.button('RUN DETECTION', on_click=run_detection)

uploaded_files = st.file_uploader("Please upload your images for detection", accept_multiple_files=True, type=['png', 'jpg'])
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        st.write("filename:", uploaded_file.name)
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.image(bytes_data)



