import streamlit as st
from utils import interactive_image

st.set_page_config(page_title="Interprétation", layout="wide")


st.title("Interprétation")

st.subheader("Analyse des erreurs")
st.write("""
La classe 'Opacité pulmonaire' introduit beaucoup de confusion. Les erreurs sont rares sur 'COVID' et 'Normal'.
""")

st.subheader("Grad-CAM et interprétabilité")
st.write("""
Visualisation des zones activées par le modèle. Bonnes correspondances avec les zones pulmonaires atteintes.

L’image ci-dessous montre une carte Grad-CAM produite par EfficientNet pour une image classée comme COVID. 
On observe une activation marquée dans la région inférieure gauche du poumon — typique des atteintes liées à la COVID-19.

Attention toutefois : l’interprétation de ces cartes d’activation reste complexe dans un contexte médical.
""")

st.image("src/images/gradcam_covid.png", caption="Activation Grad-CAM sur une image classée comme COVID", width=400)
