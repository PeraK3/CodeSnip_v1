import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import codesnip_v1

from skimage import io, color
from glob import glob
from pathlib import Path
from PIL import Image

# All imports ^^^^^^^^^^^^^^^^^^^^^

st.title(":green[Gel Band Analyzer]")

st.divider()

img_raw = st.file_uploader("Upload your gel band image", ["jpg", "jpeg", "png"]) # Uploads the image

st.divider()

df = codesnip_v1.analyze_image_V2(io.imread(img_raw)) # Analyzes the image and gives a dataframe

bg, fg= codesnip_v1.auto_thresholds(io.imread(img_raw, as_gray=True))


labels = codesnip_v1.watershed_seg_V4(image=io.imread(img_raw, as_gray=True), sure_fg=fg, sure_bg=bg)

fig, ax = plt.subplots()
im = ax.imshow(labels, cmap="nipy_spectral")
ax.set_title("Identifiable bands")

fig.colorbar(im, ax=ax) 
# Shows the area of the bands ^^^^^^^^^^^^^^^^^^^^

col3, col4 = st.columns(2)

with col3:
    st.image(img_raw)

with col4:
    st.pyplot(fig)

st.dataframe(df)

st.divider()

col1, col2 = st.columns(2)

with col1:
    sure_bg = st.color_picker("Set the background color of the image: ")
    sure_fg = st.color_picker("Set the foreground color of the image: ", "#FFFFFF" )

with col2:
    tolerance = st.slider("Band labelling tolerance", 0.0, 1.0, 0.30) # Labels whether the bands are fused or not
    min_threshold = st.slider("Delete small bands of size: ", 0, 100, 10)  # Value that removes small bands of a defined area


df_filtered = codesnip_v1.remove_redundancy(df, min_threshold)  # Removes small bands of a defined area

mean, median, mode, std = codesnip_v1.find_parameters(df_filtered)

mean = float(mean)
median = float(median)
mode = float(mode)
std = float(std)



parameters_dict = {
    "Mean":mean, 
    "Median":median, 
    "Mode":mode
}

selected_key = st.selectbox("Pick a parameter: ", parameters_dict)

selected_param = parameters_dict[selected_key]

df_filtered = codesnip_v1.classify_bands(df_filtered, selected_param)

st.divider()

st.dataframe(df_filtered) # Shows the modified dataframe

st.scatter_chart(df_filtered, x="centroid-1", y="centroid-0", x_label="X axis", y_label="Y axis", color= 'label', size = "area")



