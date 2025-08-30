import streamlit as st # type: ignore

# This the required page for the navigation
st.set_page_config(page_icon=":bar_chart:")

st.navigation({
    "Home": [
        st.Page("pages/main.py", title="Data Science Approaches for Heart Disease Prediction: A Comparative Study of Classification Algorithms"),
    ],
    "Resources": [
        st.Page("pages/about.py", title="About"),
        st.Page("pages/contact.py", title="GitHub Repo"),
    ],
}).run()