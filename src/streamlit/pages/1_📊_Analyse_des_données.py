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

palette_bar = {
    'Normal': 'green',
    'Opacité Pulmonaire': 'orange',
    'COVID-19': 'red',
    'Pneumonie virale': 'blue'}
classes_order = ['Normal', 'Opacité Pulmonaire', 'COVID-19', 'Pneumonie virale']


st.subheader("Exploration visuelle")

st.write("""
L'inspection visuelle de quelques images met en évidence que les radios sont dans l’ensemble de très bonne qualité.
""")
interactive_image("src/images/InspectionVisuelle.png", "exemple")


st.subheader("Description du jeu de données")
st.write("""
Le jeu de données comprend 21 164 images réparties entre quatre classes : Normal (10192), Opacité pulmonaire (6012), COVID-19 (3615), Pneumonie virale (1345). Les images proviennent de différentes sources médicales internationales.  
La distribution est inégale, avec 48% de radios normales et seulement 6% de pneumonies virales, ce qui peut poser des défis pour l'apprentissage automatique.
""")

df_dist = pd.DataFrame({
    'Classe': ['Normal', 'Opacité Pulmonaire', 'COVID-19', 'Pneumonie virale'],
    'Nombre d\'images': [10192, 6012, 3615, 1345]
})

st.write("""Ci-dessous une réprésentation de la répartition des classes dans le jeu de données :""")

fig = px.bar(
    df_dist,
    x='Classe',
    y="Nombre d'images",
    text="Nombre d'images",
    color='Classe',
    color_discrete_map=palette_bar)
fig.update_traces(textposition='outside')
fig.update_layout(
    xaxis_title="Classe",
    yaxis_title="Nombre d'images",
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Distribution de la variance par classe")


st.write("""Ci-dessous une réprésentation de la variance par classe (plus la variance est élevée, plus l'image est complexe/texturée) :""")


script_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.dirname(script_dir)    
input_filename = os.path.join(project_root, 'data', 'variance.csv')    
df_plot = pd.read_csv(input_filename)  


fig = px.violin(
    df_plot,
    x='classe',
    y='variance',
    color='classe',
    color_discrete_map=palette_bar,
    box=True,  
    points=False,
    category_orders={'classe': classes_order},
    labels={
        'classe': 'Classe de la radiographie',
        'variance': 'Variance des pixels'
    }
)
fig.update_layout(
    xaxis_title="Catégorie",
    yaxis_title="Variance",
    legend_title="Légende"
)
st.plotly_chart(fig, use_container_width=True)

# Remplacé par graphique interactif plotly en violin plot
# interactive_image("src/images/Variance.png", "exemple")

st.subheader("Détection d'anomalies")
st.write("""
Des anomalies ont été identifiées, telles que des doublons ou des images de faible qualité.
""")

st.write("""
Avec la méthode IQR sur l’intensité et l’écart-type, après normalisation de l’intensité (seuil à 1,5 x IQR) : 285 outliers identifiés.  
Ci-dessous une visualisation de la répartition de l’intensité en fonction de l’écart-type sur les radios après normalisation :
""")


script_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.dirname(script_dir)    
input_filename = os.path.join(project_root, 'data', 'iqr_outliers.csv')    
df_plot = pd.read_csv(input_filename)  

taille_mapping = {'Non': 5, 'Oui': 20} 
df_plot['taille_point'] = df_plot['est_outlier'].map(taille_mapping)

fig = px.scatter(
    df_plot,
    x='intensite_moyenne',
    y='ecart_type',
    color='classe',
    color_discrete_map=palette_bar,
    symbol='est_outlier',
    size='taille_point',         
    #symbol_map={'Non': 'circle', 'Oui': 'star-diamond'},
    category_orders={
        'classe': classes_order,
        'est_outlier': ['Non', 'Oui']
    },
    labels={
        'intensite_moyenne': 'Intensité Moyenne (Image Normalisée)',
        'ecart_type': 'Écart-Type des Pixels (Contraste/Texture)',
        'classe': 'Classe de la Radiographie',
        'est_outlier': 'Est un Outlier ?'
    },
    title='Distribution des Radiographies par Intensité et Écart-Type'
)

fig.update_traces(hoverinfo='none', hovertemplate=None)
fig.update_layout(
    legend_title="Légendes",
    height=700 
)

st.plotly_chart(fig, use_container_width=True)


interactive_image("src/images/Intensite-ecart.png", "exemple")


st.write("#### Elimination des anomalies")

DESCRIPTIONS = {
    'Statistique': "Cette méthode fondamentale transforme chaque image en caractéristiques numériques (moyenne, contraste, entropie). Le score d'anomalie est basé sur la distance d'une image à la distribution normale. La technique fonctionne très bien pour trouver les images très sombres ou vides.",
    'Isolation Forest': "Cette approche utilise un réseau de neurones VGG16 entraîné sur des images pour extraire des caractéristiques complexes. L'algorithme Isolation Forest isole ensuite les images qui sont sémantiquement différentes des autres. Cette approche est efficace pour trouver des textures ou des formes inhabituelles (présence de colliers, pacemakers, etc.)",
    'Auto-encoder': "Un réseau de neurones est entraîné à compresser puis reconstruire les images du dataset. Il devient expert des radiographies 'typiques'. Une image qu'il peine à reconstruire (erreur élevée) est considérée comme anormale. C'est l'approche la plus sensible aux anomalies subtiles."
}

options = ["Statistique", "Isolation Forest", "Auto-encoder"]
selection = st.segmented_control(
    "",
    options,
    label_visibility="collapsed", 
    default="Statistique"
)

if selection == "Statistique":
    st.write("#### Approche Statistique")
    st.write(DESCRIPTIONS[selection])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.dirname(script_dir)    
    input_filename = os.path.join(project_root, 'data', 'plot_data_statistique.csv')
    df_plot = pd.read_csv(input_filename)
    fig = px.scatter_3d(
            df_plot,
            x='Moyenne Normalisée',
            y='Écart-type Normalisé',
            z='Entropie Normalisée',
            color='score',
            size='score',
            size_max=20,
            color_continuous_scale=px.colors.sequential.Viridis,
        )
    fig.update_traces(marker=dict(opacity=0.8), hoverinfo='none', hovertemplate=None)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.write("Les 10 images les plus anormales trouvées :")

    method_key = selection.lower().replace(' ', '_')  
    for i in range(2):
        cols = st.columns(5)
        for j in range(5):
            rank = i * 5 + j + 1
            script_dir = os.path.dirname(os.path.abspath(__file__))    
            project_root = os.path.dirname(script_dir)    
            image_path = os.path.join(project_root, 'outliers_images', f"{method_key}_anomaly_{rank}.png")
            with cols[j]:
                try:
                    st.image(Image.open(image_path), use_container_width=True)
                except FileNotFoundError:
                    st.markdown(f"_(Image #{rank} non trouvée)_")    

elif selection == "Isolation Forest":
    st.write("#### Approche Machine Learning (Isolation Forest)")
    st.write(DESCRIPTIONS[selection])

    script_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.dirname(script_dir)    
    input_filename = os.path.join(project_root, 'data', 'plot_data_isolation_forest.csv')
    df_plot = pd.read_csv(input_filename)
    
    score_values = df_plot['score'].values
    min_val, max_val = score_values.min(), score_values.max()
    df_plot['size_score'] = (score_values - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0

    fig = px.scatter_3d(
        df_plot,
        x='PC1', y='PC2', z='PC3',
        color='score',
        size='size_score',
        size_max=20,
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig.update_traces(marker=dict(opacity=0.8), hoverinfo='none', hovertemplate=None)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.write("Les 10 images les plus anormales trouvées :")


    method_key = selection.lower().replace(' ', '_')
    for i in range(2):
        cols = st.columns(5)
        for j in range(5):
            rank = i * 5 + j + 1
            script_dir = os.path.dirname(os.path.abspath(__file__))    
            project_root = os.path.dirname(script_dir)    
            image_path = os.path.join(project_root, 'outliers_images', f"{method_key}_anomaly_{rank}.png")
            with cols[j]:
                try:
                    st.image(Image.open(image_path), use_container_width=True)
                except FileNotFoundError:
                    st.markdown(f"_(Image #{rank} non trouvée)_")   

elif selection == "Auto-encoder":
    st.write("#### Approche Deep Learning (Auto-encodeur)")
    st.write(DESCRIPTIONS[selection])

    script_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.dirname(script_dir)    
    input_filename = os.path.join(project_root, 'data', 'plot_data_autoencoder.csv')
    df_plot = pd.read_csv(input_filename)

    fig = px.scatter_3d(
        df_plot,
        x='Latent PC1', y='Latent PC2', z='Latent PC3',
        color='score',
        size='score',
        size_max=20,
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig.update_traces(marker=dict(opacity=0.8), hoverinfo='none', hovertemplate=None)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.write("Les 10 images les plus anormales trouvées :")
    
    method_key = selection.lower().replace(' ', '_').replace('-', '')
    for i in range(2):
        cols = st.columns(5)
        for j in range(5):
            rank = i * 5 + j + 1
            script_dir = os.path.dirname(os.path.abspath(__file__))    
            project_root = os.path.dirname(script_dir)    
            image_path = os.path.join(project_root, 'outliers_images', f"{method_key}_anomaly_{rank}.png")
            with cols[j]:
                try:
                    st.image(Image.open(image_path), use_container_width=True)
                except FileNotFoundError:
                    st.markdown(f"_(Image #{rank} non trouvée)_")   



st.subheader("Réductions de dimensions")

options = ["PCA", "UMAP", "AE", "NMF"]
selection = st.segmented_control("", 
                                 options, 
                                 selection_mode="single",
                                 default="PCA"
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

st.write("Plusieurs techniques de prétraitement ont été appliquées durant le projet. Ces étapes se sont avérées très utiles pour améliorer la performance des modèles de machine learning.")

st.write("#### Redimensionnement, normalisation et amélioration du contraste")

st.write("""Les images ont été redimensionnées à 240 x 240 pixels et normalisées. En effet, il a été constaté que 7 radiographies sur 10 ne sont pas normalisées. L'amélioration du contraste par la technique CLAHE a également été testée. Elle améliore le contraste en l'égalisant sur de petites zones locales de l'image, ce qui permet de rehausser les détails fins sans sur-amplifier le bruit de manière globale. Sur les radiographies, cette technique est très pertinente car elle fait ressortir les structures subtiles des tissus mous sans saturer les zones très denses comme les os, améliorant ainsi la visibilité pour le diagnostic.
""")

@st.cache_data 
def get_image_paths(folder):
    if not os.path.isdir(folder):
        return []
    supported_extensions = ('.png', '.jpg', '.jpeg')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(supported_extensions)]

def transform_image_randomly(pil_image):
    image_np = np.array(pil_image.convert('L'))
    if random.random() < 0.5: image_np = cv2.flip(image_np, 1) 
    h, w = image_np.shape
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    tx = random.uniform(-w * 0.05, w * 0.05)
    ty = random.uniform(-h * 0.05, h * 0.05)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)    
    M[0, 2] += tx
    M[1, 2] += ty    
    transformed_image = cv2.warpAffine(image_np, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    transformed_image = clahe.apply(transformed_image)
    normalized_image = transformed_image.astype(np.float32) / 255.0    
    return normalized_image


script_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_dir = os.path.dirname(script_dir) 
IMAGE_DIR = os.path.join(streamlit_dir, 'images')

all_image_paths = get_image_paths(IMAGE_DIR)

st.write("#### Augmentation de Données")

st.write("Les images ont été enrichies par augmentation de données (retournements horizontaux, rotations, zooms, translations aléatoires).")

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

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        if st.button("🔄 Nouvelle image", use_container_width=True):
            st.session_state.current_image_path = random.choice(all_image_paths)
            st.session_state.transformed_image = None
            st.rerun()

    with control_col2:
        is_disabled = (original_image is None)
        if st.button("✨ Transformation", use_container_width=True, disabled=is_disabled):
            if original_image:
                st.session_state.transformed_image = transform_image_randomly(original_image)
                
    image_col1, image_col2 = st.columns(2)

    with image_col1:
        st.markdown("<h5 style='text-align: center;'>Image Originale</h5>", unsafe_allow_html=True)
        if original_image:
            left_space, img_container, right_space = st.columns([1, 2, 1])
            with img_container:
                file_name = os.path.basename(st.session_state.current_image_path)
                st.image(original_image, caption=f"Fichier : {file_name}", width=300) 

    with image_col2:
        st.markdown("<h5 style='text-align: center;'>Image Transformée</h5>", unsafe_allow_html=True)
        if st.session_state.transformed_image is not None:
            left_space, img_container, right_space = st.columns([1, 2, 1])
            with img_container:
                st.image(st.session_state.transformed_image, caption="Résultat de l'augmentation", width=300)
        else:
            st.info("L'image transformée apparaitra ici.")
            st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)

