import streamlit as st
import pandas as pd
import plotly.express as px
from utils import interactive_image
import os
from PIL import Image
import cv2  
import random
import numpy as np

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
La distribution est inégale, avec 48% de radios normales et seulement 6% de pneumonies virales, ce qui peut poser des défis pour l'apprentissage automatique.
""")


df_dist = pd.DataFrame({
    'Classe': ['Normal', 'Opacité Pulmonaire', 'COVID-19', 'Pneumonie virale'],
    'Nombre d\'images': [10192, 6012, 3615, 1345]
})


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

st.write("""
Visualisation statistique : variance de l’intensité, projections UMAP, et examen manuel sur quelques images.
Variance : ci-dessous une visualisation de la variance par classe
""")
interactive_image("src/images/Variance.png", "exemple")

st.subheader("Détection d'anomalies")
st.write("""
Des anomalies ont été identifiées, telles que des doublons ou des images de faible qualité.
""")

st.write("""
Avec la méthode IQR sur l’intensité et l’écart-type, après normalisation de l’intensité (seuil à 1,5 x IQR) : 285 outliers identifiés.  
Ci-dessous une visualisation de la répartition de l’intensité en fonction de l’écart-type sur les radios après normalisation :
""")
interactive_image("src/images/Intensite-ecart.png", "exemple")



st.subheader("Réductions de dimensions")



options = ["PCA", "AE", "NMF", "UMAP"]
selection = st.segmented_control("", options, selection_mode="single"
)

palette = {
    'Normal': 'green',
    'COVID': 'red',
    'Lung_Opacity': 'orange',
    'Viral Pneumonia': 'blue'}

if selection == "PCA" :
    st.write("#### Analyse en Composantes Principales (PCA)")
    st.write("PCA trouve de nouveaux axes (appelés composantes principales) qui maximisent la variance (la dispersion) des données. Le premier axe capture le plus de variance possible, le deuxième en capture le plus possible parmi ce qu'il reste, et ainsi de suite. C'est une méthode purement mathématique et linéaire.")
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.dirname(script_dir)    
    input_filename = os.path.join(project_root, 'data', 'pca_3d_data.csv')    
    df_plot = pd.read_csv(input_filename)  
    fig = px.scatter_3d(
        df_plot,
        x='PCA 1',
        y='PCA 2',
        z='PCA 3',
        color='label', 
        title="Visualisation 3D PCA des radiographies pulmonaires",
        color_discrete_map=palette,
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(legend_title_text='Classe',margin=dict(l=0, r=0, b=0, t=0))
    fig.update_traces(hoverinfo='none', hovertemplate=None)
    st.plotly_chart(fig, use_container_width=True)

if selection == "UMAP" :
    st.write("#### Uniform Manifold Approximation and Projection (UMAP)")
    st.write("UMAP est une technique d'apprentissage de variétés (manifold learning). Elle suppose que les données, même si elles sont dans un grand espace, vivent en réalité sur une surface de plus faible dimension (la variété). UMAP essaie de modéliser cette surface et de la déplier dans un espace plus petit tout en préservant au mieux la structure topologique des données (qui est voisin de qui, localement et globalement).")
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.dirname(script_dir)    
    input_filename = os.path.join(project_root, 'data', 'umap_3d_data.csv')    
    df_plot = pd.read_csv(input_filename)  
    fig = px.scatter_3d(
        df_plot,
        x='UMAP 1',
        y='UMAP 2',
        z='UMAP 3',
        color='label', 
        title="Visualisation 3D UMAP des radiographies pulmonaires",
        color_discrete_map=palette,
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(legend_title_text='Classe',margin=dict(l=0, r=0, b=0, t=0))
    fig.update_traces(hoverinfo='none', hovertemplate=None)
    st.plotly_chart(fig, use_container_width=True)

    
elif selection == "AE" :
    st.write("#### Auto-Encoder (AE)")
    st.write("C'est un type de réseau de neurones qui apprend à compresser les données (partie encodeur) en une représentation de faible dimension (le goulot d'étranglement ou bottleneck), puis à les décompresser (partie décodeur) pour reconstruire l'entrée originale. En forçant le réseau à recréer les données à partir d'une version compressée, il apprend les caractéristiques les plus importantes.")
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.dirname(script_dir)    
    input_filename = os.path.join(project_root, 'data', 'ae_3d_data.csv')    
    df_plot = pd.read_csv(input_filename)  
    fig = px.scatter_3d(
        df_plot,
        x='AE 1',
        y='AE 2',
        z='AE 3',
        color='label', 
        title="Visualisation 3D AE des radiographies pulmonaires",
        color_discrete_map=palette,
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(legend_title_text='Classe',margin=dict(l=0, r=0, b=0, t=0))
    fig.update_traces(hoverinfo='none', hovertemplate=None)
    st.plotly_chart(fig, use_container_width=True)

    
if selection == "NMF" :
    st.write("#### Non-negative Matrix Factorization (NMF)")
    st.write("La NMF décompose une grande matrice de données (par exemple, des images ou des documents) en deux matrices plus petites. La contrainte essentielle est que toutes les valeurs dans les trois matrices doivent être non-négatives. Cela force la décomposition à être additive.")
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.dirname(script_dir)    
    input_filename = os.path.join(project_root, 'data', 'nmf_3d_data.csv')    
    df_plot = pd.read_csv(input_filename)  
    fig = px.scatter_3d(
        df_plot,
        x='NMF 1',
        y='NMF 2',
        z='NMF 3',
        color='label', 
        title="Visualisation 3D NMF des radiographies pulmonaires",
        color_discrete_map=palette,
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(legend_title_text='Classe',margin=dict(l=0, r=0, b=0, t=0))
    fig.update_traces(hoverinfo='none', hovertemplate=None)
    st.plotly_chart(fig, use_container_width=True)



st.subheader("Prétraitement")
st.write("""
Les images ont été redimensionnées à 240x240 pixels, normalisées, et enrichies par augmentation de données (flip, rotation, zoom). Des méthodes comme Isolation Forest ont été utilisées pour retirer les outliers.

Il a été constaté que 7 radiographies sur 10 ne sont pas normalisées. Voici la représentation en fonction des diverses sources de données initiales :            
""")


@st.cache_data 
def get_image_paths(folder):
    """Scanne un dossier et retourne la liste des chemins des fichiers image valides."""
    if not os.path.isdir(folder):
        return []
    supported_extensions = ('.png', '.jpg', '.jpeg')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(supported_extensions)]

def transform_image_randomly(pil_image):
    image_np = np.array(pil_image.convert('L'))
    h, w = image_np.shape
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    tx = random.uniform(-w * 0.05, w * 0.05)
    ty = random.uniform(-h * 0.05, h * 0.05)
    M_rot_zoom = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)    
    M_rot_zoom[0, 2] += tx
    M_rot_zoom[1, 2] += ty
    transformed_image = cv2.warpAffine(image_np, M_rot_zoom, (w, h), borderMode=cv2.BORDER_REPLICATE)
    normalized_image = transformed_image.astype(np.float32) / 255.0    
    return normalized_image



script_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_dir = os.path.dirname(script_dir) 
IMAGE_DIR = os.path.join(streamlit_dir, 'images')

all_image_paths = get_image_paths(IMAGE_DIR)

st.write("🔬 Démonstration de l'Augmentation de Données")

if not all_image_paths:
    st.error(f"Aucune image trouvée dans le dossier '{IMAGE_DIR}'.")
    st.warning("Veuillez vérifier que le dossier existe et contient des images.")
else:
    if 'current_image_path' not in st.session_state or st.session_state.current_image_path not in all_image_paths:
        st.session_state.current_image_path = random.choice(all_image_paths)
        st.session_state.transformed_image = None   

    try:
        original_image = Image.open(st.session_state.current_image_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image : {e}")
        original_image = None 

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image Originale")
        
        if st.button("🔄 Nouvelle image", use_container_width=True):
            st.session_state.current_image_path = random.choice(all_image_paths)
            st.session_state.transformed_image = None
            st.rerun() 
        
        if original_image:
            if st.button("✨ Transformation", use_container_width=True, type="primary"):
                st.session_state.transformed_image = transform_image_randomly(original_image)
        
        if original_image:
            file_name = os.path.basename(st.session_state.current_image_path)
            st.image(original_image, caption=f"Fichier : {file_name}", use_column_width=True)

    with col2:
        st.subheader("Image Transformée")
        if st.session_state.transformed_image is not None:
            st.image(
                st.session_state.transformed_image,
                caption="Transformation + Normalisation",
                use_column_width=True
            )
        else:
             st.info("Cliquez sur 'Transformation' pour générer une version augmentée.")


interactive_image("src/images/Normalisation.png", "exemple")
