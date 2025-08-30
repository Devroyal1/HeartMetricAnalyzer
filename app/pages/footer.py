import streamlit as st  # type: ignore

def footer():
    st.html('''
    <footer style="
    text-align: center;
    padding: 10px;
    font-size: 12px;
    width: 100%;
    clear: both;">
        <p style="margin-bottom: 5px;">Created by <a href="https://www.linkedin.com/in/Devendranath-Bhavanasi/" target="_blank" style="color: #66d9ef;">Devendranath Bhavanasi</a></p>
        <p style="margin-bottom: 5px;">View on <a href="https://github.com/Devroyal1/HeartMetricAnalyzer" target="_blank" style="color: #66d9ef;">GitHub</a></p>
    </footer>
    ''')