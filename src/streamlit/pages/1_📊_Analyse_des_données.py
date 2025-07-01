import streamlit as st
import pandas as pd
import plotly.express as px
from utils import interactive_image

st.set_page_config(page_title="Analyse des Données", layout="wide")

st.title("Analyse des données")


st.subheader("Exploration visuelle")

st.write("""
Inspection visuelle de quelques images : l’inspection visuelle met en évidence que les radios sont dans l’ensemble de très bonne qualité.
""")
interactive_image("src/images/InspectionVisuelle.png", "exemple")


st.subheader("Description du jeu de données")
st.write("""
Le jeu de données comprend 21 164 images réparties entre quatre classes : Normal (10192), COVID-19 (3615), Pneumonie virale (1345), Opacité pulmonaire (6012). Les images proviennent de différentes sources médicales internationales.
""")

st.subheader("Distribution des classes")
df_dist = pd.DataFrame({
    'Classe': ['Normal', 'Opacité Pulmonaire', 'COVID-19', 'Pneumonie virale'],
    'Nombre d\'images': [10192, 6012, 3615, 1345]
})

st.table(df_dist.set_index("Classe"))

fig = px.bar(
    df_dist,
    x='Classe',
    y="Nombre d'images",
    text="Nombre d'images",
    color='Classe',
    title="Répartition des classes dans le jeu de données",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='outside')
fig.update_layout(
    xaxis_title="Classe",
    yaxis_title="Nombre d'images",
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Analyse exploratoire")
st.write("""
PCA, UMAP, autoencodeurs et histogrammes ont permis de visualiser la structure latente du jeu de données. Des anomalies ont été identifiées, telles que des doublons ou des images de faible qualité.

La distribution est inégale, avec 48% de radios normales et seulement 6% de pneumonies virales, ce qui peut poser des défis pour l'apprentissage automatique.
""")
interactive_image("src/images/DistributionClasses.png", "exemple")

st.write("""
Visualisation statistique : variance de l’intensité, projections UMAP, et examen manuel sur quelques images.
Variance : ci-dessous une visualisation de la variance par classe
""")
interactive_image("src/images/Variance.png", "exemple")

st.write("Intensité vs. écart-type : ci-dessous une visualisation de la répartition de l’intensité en fonction de l’écart-type sur les radios après normalisation :")
interactive_image("src/images/Intensite-ecart.png", "exemple")



st.subheader("Réductions de dimensions")

options = ["PCA", "UMAP", "AE", "NMF"]
selection = st.segmented_control("", options, selection_mode="single"
)
if selection == "PCA" :
    st.write("#### Analyse en Composantes Principales (PCA)")
    st.write("""
             Projection 2D après réduction de dimension via PCA (linéaire) et normalisation préalable
             """)
    interactive_image("src/images/Projection2d.png", "exemple")
elif selection == "UMAP" :
    st.write("#### UMAP : Uniform Manifold Approximation and Projection")
    st.write("""
    Projection 2D après réduction de dimension via UMAP non linéaire (High Performance Dimension Reduction) et normalisation préalable:
    """)
    interactive_image("src/images/UMAP.png", "exemple")
elif selection == "AE" :
    st.write("#### Auto-Encoder (AE)")
    st.write("""
    Projection 2D après encodage / décodage par auto-encodeur (AE) et normalisation préalable
    """)
    interactive_image("src/images/Autoencoder.png", "exemple")
if selection == "NMF" :
    st.write("#### NMF : Non-negative Matrix Factorization")
    st.write("""
    Projection 2D après encodage / décodage par NMF (à ajouter) et normalisation préalable
    """)
    interactive_image("src/images/Autoencoder.png", "exemple")



st.subheader("Prétraitement")
st.write("""
Les images ont été redimensionnées à 240x240 pixels, normalisées, et enrichies par augmentation de données (flip, rotation, zoom). Des méthodes comme Isolation Forest ont été utilisées pour retirer les outliers.

Il a été constaté que 7 radiographies sur 10 ne sont pas normalisées. Voici la représentation en fonction des diverses sources de données initiales :            
""")
interactive_image("src/images/Normalisation.png", "exemple")
