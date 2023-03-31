import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np


def gen_settings() -> list:
    settings = []
    settings.append(st.sidebar.checkbox("Verschärfen"))
    settings.append(st.sidebar.checkbox("Filter"))
    settings.append(st.sidebar.slider("Hue", min_value=0, max_value=100))
    settings.append(st.sidebar.slider("Chroma", min_value=-50, max_value=50, value=0))
    settings.append(st.sidebar.slider("Lightness", min_value=-50.0, max_value=100.0, step=0.01, value=0.0))
    settings.append(st.sidebar.slider("Contrast", min_value=0.0, max_value=10.0, step=0.1))
    return settings


def displayU(image, edited=False) -> None:
    if not edited:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image
    edited_img = img

    settings = gen_settings()

    setting_sharp = settings[0]
    setting_filter = settings[1]
    setting_hue = settings[2]
    setting_chroma = settings[3]
    setting_lightness = settings[4]
    setting_contrast = settings[5]

    if setting_sharp:
        sharp_value = setting_sharp
    else:
        sharp_value = 0

    if setting_filter:
        filter_value = setting_filter
    else:
        filter_value = 0

    if setting_hue:
        st.write("hue")
        hue_value = setting_hue
    else:
        hue_value = 0

    if setting_chroma:
        chroma_value = setting_chroma
    else:
        chroma_value = 0

    if setting_lightness:
        lightness_value = setting_lightness
    else:
        lightness_value = 0

    if setting_contrast:
        set_contrast = setting_contrast
    else:
        set_contrast = 1

    if sharp_value is True:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        edited_img = cv2.filter2D(edited_img, -1, kernel)
    
    if filter_value is True:
        if st.sidebar.checkbox("Median"):
            edited_img = cv2.medianBlur(edited_img, 3)
        if st.sidebar.checkbox("Laplace"):
            edited_img = cv2.Laplacian(edited_img, -1)
        if st.sidebar.checkbox("Unsharp Masking"):
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            edited_img = cv2.filter2D(edited_img, -1, kernel)
        if st.sidebar.checkbox("Erode"):
            kernel = np.ones((5, 5), np.uint8)
            edited_img = cv2.erode(edited_img, kernel, iterations=1)
        if st.sidebar.checkbox("Dilate"):
            kernel = np.ones((5, 5), np.uint8)
            edited_img = cv2.dilate(edited_img, kernel, iterations=1)
        
    if hue_value is not 0:
        hsv = cv2.cvtColor(edited_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        h = h + hue_value
        final_hsv = cv2.merge((h, s, v))
        edited_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    if chroma_value is not 0:
        hsv = cv2.cvtColor(edited_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = np.int16(s) + chroma_value
        s = np.clip(s, 0, 255).astype(np.uint8)
        final_hsv = cv2.merge((h, s, v))
        edited_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        

    if lightness_value is not 0:
        lab = cv2.cvtColor(edited_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_norm = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        l_adjusted = cv2.addWeighted(l_norm, 1, l_norm, 0, lightness_value - 50)
        lab_adjusted = cv2.merge((l_adjusted, a, b))
        edited_img = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2RGB)

    if set_contrast is not 1:
        edited_img = cv2.addWeighted(edited_img, set_contrast, np.zeros(edited_img.shape, edited_img.dtype), 0, 0)

    st.image(edited_img, use_column_width=True)
    return edited_img


def editor() -> None:
    image = None
    edited_img = None
    ed = st.sidebar.checkbox("Editiert")
    if not ed:
        upload = st.selectbox("Kamera oder Upload", ["Kamera", "Upload"])

        if upload == "Upload":
            image = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])
        else:
            img_file_buffer = st.camera_input("")
            if img_file_buffer is not None:
                image = img_file_buffer
    else:
        image = cv2.imread("storage/tmp/temp.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.sidebar.markdown("---")

    if image is not None:
        edited_img = displayU(image, ed)
        st.session_state["edited_img"] = edited_img

        if st.sidebar.button("Speichern"):
            plt.imsave("storage/tmp/temp.jpg", edited_img)
