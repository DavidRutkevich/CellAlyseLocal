import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
def prepare_upload(img, cam=False):
    if not cam:
        image = Image.open(img)
        image.save("storage/tmp/temp.jpg")
    else:
        plt.imsave("storage/tmp/temp.jpg", img)
        return None
    return img
