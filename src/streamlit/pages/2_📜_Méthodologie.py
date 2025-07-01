import streamlit as st
from utils import interactive_image

st.set_page_config(page_title="Méthodologie", layout="wide")

st.title("Méthodologie")

st.subheader("Classification du problème")
st.write("""
Il s’agit d’un problème de classification multi-classes supervisée, relevant du
domaine de la santé et plus précisément du diagnostic radio assisté par IA.
""")

st.subheader("Pipeline général")
st.markdown("""
1. **Chargement et prétraitement des images**  
2. **Échantillonnage équilibré ou pondération des classes dans les modèles**  
3. **Entraînement d’algorithmes de machine learning / deep learning**  
4. **Optimisation des modèles et des hyperparamètres**  
5. **Évaluation croisée, analyse d’erreurs, interprétation visuelle**
""")