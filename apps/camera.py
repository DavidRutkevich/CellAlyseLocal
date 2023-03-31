import cv2
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional, Any
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from streamlit_cropper import st_cropper

from helper.svm import large_image, bbox, segmentation, read, predict_svm
from helper.functions import prepare_upload
from helper.model import (
    st_predict,
    stcount,
    hough_transform,
    component_labeling,
    stthreshold,
    output_directory,
)


def gen_settings(celltype_options: List[str], auswahl: set) -> Tuple[List[str], List[bool]]:
    celltype = st.sidebar.multiselect("Zelltypen auswählen", celltype_options)
    settings = []

    if set(celltype) & auswahl:
        st.sidebar.markdown("---")
        cht = st.sidebar.checkbox("Circle Hough Transform")
        ccl = st.sidebar.checkbox("Connected Component Labeling")
        dt = st.sidebar.checkbox("Distance Transform")
        settings = [cht, ccl, dt]

    if "Weiße Blutzellen" in celltype:
        st.sidebar.markdown("---")
        model = st.sidebar.radio(
            "Modell auswählen", ("Raabin", "LISC", "BCCD"), index=0
        )
        bound = st.sidebar.checkbox("Bounding Boxes")
        settings.append(bound)
        settings.append(model)

    st.sidebar.markdown("---")
    settings.append(st.sidebar.checkbox("Analyse starten"))
    return celltype, settings


@st.cache_data
def wbc_iz(image: np.ndarray, model: str, bb: bool) -> None:
    if bb:
        out = bbox(image)
        st.image(out)
    else:
        nucl, _, _ = segmentation(image)
        st.image(nucl)
    large_image(image, model)

@st.cache_data
def tpyisit(image: np.ndarray, celltype: str, settings: List[bool]) -> np.ndarray:
    prepare_upload(image, True)
    image = st_predict(f"{output_directory}/temp.jpg", cell_type=celltype)
    st.image(image, use_column_width=True, clamp=True)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if "Rote Blutzellen" in celltype:
        out_image = "edge_mask.png"
    else:
        out_image = "mask.png"

    image = cv2.normalize(
        src=image,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    if settings[0]:
        hough_transform(image, celltype)
    if settings[1]:
        component_labeling(image)
    if settings[2]:
        threshold = stthreshold(image, celltype)
        threshold = cv2.normalize(src=threshold, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)
        stcount(threshold, celltype)


def camera() -> Optional[Any]:
    img_file_buffer = st.camera_input("")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    return None

def read_direction_file():
    file_path = Path("storage/move.txt")
    if not file_path.exists():
        with open(file_path, "w") as f:
            f.write("0")
    with open(file_path, "r") as f:
        direction = f.read()
    return direction

def write_direction_file(direction):
    file_path = Path("storage/move.txt")
    with open(file_path, "w") as f:
        f.write(str(direction))

def move_CellAlyzer():
    direction = read_direction_file()

    

    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("")
        with col2:
            if st.button("⬆️"):
                direction = "up"
                write_direction_file(direction)
        with col3:
            st.write("")


        col4, col5, col6 = st.columns(3)

        with col4:
            if st.button("⬅️"):
                direction = "left"
                write_direction_file(direction)

        with col5:
            if st.button("⬇️"):
                direction = "down"
                write_direction_file(direction)

        with col6:
            if st.button("➡️"):
                direction = "right"
                write_direction_file(direction)
        
def button(*args, key=None, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if "type" not in kwargs:
        kwargs["type"] = "primary" if st.session_state[key] else "secondary"

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]
        st.experimental_rerun()

    return st.session_state[key]

def entry() -> None:
    cell_types = ["Rote Blutzellen", "Weiße Blutzellen", "Plättchen"]
    selected_cell_types = {"Rote Blutzellen", "Plättchen"}
    
    celltype, settings = gen_settings(cell_types, selected_cell_types)
    
    edited = st.sidebar.checkbox("Edited")

    if not edited:
        cam_or_upload = st.selectbox("Kamera oder Upload", ["Kamera", "Upload"], key=0o6)
        if cam_or_upload == "Kamera":
            image = camera()
        else:
            upload = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])
            if upload is not None:
                image = Image.open(upload)
                image.save("storage/tmp/temp.jpg")
                image = cv2.imread("storage/tmp/temp.jpg")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image)
    else:
        image = cv2.imread("storage/tmp/temp.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image)
    
    if st.sidebar.checkbox("Zuschneiden"):
        realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
        box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
        aspect_dict = {
            "Free": None
        }
        aspect_ratio = aspect_dict["Free"]

        if image is not None:
            img = Image.fromarray(image)
            if not realtime_update:
                st.write("Double click to save crop")
            cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                        aspect_ratio=aspect_ratio)
            
            cropped_img = np.array(cropped_img)
            image = cropped_img
            st.image(image)

    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()

    if "Weiße Blutzellen" in celltype and settings[-1]:
        wbc_iz(image, settings[-2], settings[-3])

    if "Rote Blutzellen" in celltype and settings[-1]:
        tpyisit(image, "rbc", settings)

    if "Plättchen" in celltype and settings[-1]:
        tpyisit(image, "plt", settings)
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Mikroskop Bewegen"):
        move_CellAlyzer()