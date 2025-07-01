# utils.py

import streamlit as st
from PIL import Image
import plotly.graph_objects as go

def interactive_image(path, caption=None):
    # Cette fonction reste inchangée
    img = Image.open(path)
    width, height = img.size
    fig = go.Figure()
    fig.add_layout_image(
        dict(source=img, x=0, y=height, sizex=width, sizey=height, xref="x", yref="y", sizing="contain", layer="below")
    )
    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height])
    fig.update_layout(width=width, height=height, margin=dict(l=0, r=0, t=0, b=0), dragmode="zoom", title=caption if caption else "")
    st.plotly_chart(fig, use_container_width=True)