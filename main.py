import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

from apps.camera import entry
from apps.editor import editor

icon = Image.open("style/favicon.ico")
st.set_page_config(
    page_title="CellAlyse",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="expanded",
)
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
               height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Camera", "Hello"],
        icons=["house", "moisture"],
        styles={
            "icon": {"color": "#dde5e5", "font-size": "20px"},
            "nav-link-selected": {"background-color": "#1e1f2b"},
            "nav-link": {"font-size": "20px", "color": "#dde5e5"},
        },
    )

if selected == "Camera":
    entry()
if selected == "Hello":
    editor()