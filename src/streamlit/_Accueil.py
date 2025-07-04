import streamlit as st

st.set_page_config(
    page_title="Accueil - Détection pathologies pulmonaires",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded" 
)

col1, col2 = st.columns([1, 20])
with col1:
    st.image("src/images/Normal-2.png", width=170)
with col2:
    st.title("Détection de pathologies pulmonaires à partir de radiographies thoraciques")

st.write("_Projet basé sur le rapport d'analyse de radiographies pour la détection de la COVID-19 et autres pathologies._")


st.write("### Introduction")


st.write("""
Le diagnostic médical par imagerie radiologique est un élément clé de la médecine moderne. Toutefois, l’interprétation manuelle des radiographies thoraciques est souvent complexe, sujette à erreurs humaines et dépend fortement de l’expertise du radiologue.

Ce projet, réalisé dans le cadre de la formation **Data Scientist - Septembre 2024** chez Data Scientest, vise à développer un pipeline automatique de classification des radiographies pulmonaires. L’objectif principal est la détection précise de la COVID-19, de la pneumonie virale et d'autres pathologies pulmonaires.

La méthodologie comprend une exploration et un pré-traitement approfondis des données, suivi de la comparaison entre modèles classiques de Machine Learning et architectures avancées de Deep Learning, incluant des techniques de transfer learning et des modèles hybrides CNN-ViT.

Les performances des modèles sont évaluées via plusieurs métriques clés : précision, sensibilité (rappel), spécificité, F1-score pondéré et matrice de confusion.
""")

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
