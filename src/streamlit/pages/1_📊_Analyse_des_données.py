import streamlit as st
import pandas as pd
import plotly.express as px
from utils import interactive_image
import os
from PIL import Image
import cv2  
import random
import numpy as np
import plotly.graph_objects as go


st.set_page_config(page_title="Analyse des Données", layout="wide")

st.title("Analyse des données")

script_dir = os.path.dirname(os.path.abspath(__file__))    
project_root = os.path.dirname(script_dir)

palette_bar = {
    'Normal': 'green',
    'Opacité Pulmonaire': 'orange',
    'COVID-19': 'red',
    'Pneumonie virale': 'blue'}
classes_order = ['Normal', 'Opacité Pulmonaire', 'COVID-19', 'Pneumonie virale']

st.subheader("Exploration des données")

options = ["Jeu de données", "Inspection Visuelle", "Analyse statistique"]
selection = st.segmented_control("", options, selection_mode="single", default="Jeu de données"
)

if selection == "Jeu de données":
    st.write("#### Description du jeu de données")
    st.write("""
    - **~ 20 000** radios des poumons  
    - Issues de différentes **sources médicales** internationales  
    - Mis à disposition sur **Kaggle** par une équipe de chercheurs de l’université du Qatar à Doha  
    - **Base de référence**, qui a été régulièrement enrichie  
    - **Masques** associés  
    - **Taille** des images : 299x299  
    - **4 classes** : Covid, Normal, Opacité Pulmonaire, Pneumonie Virale
    """)

elif selection == "Inspection Visuelle":
    st.write("#### Exploration visuelle des images")
    st.write("""
    - Radios dans l’ensemble de très bonne qualité
    - Présence de matériel médical visible sur certaines radios
    - Annotations également souvent (L, R, ORTHO...)  
    - Quelques clichés râtés (sur/sous exposition, pb de cadrage, floues)
    """)
    st.write("Voici quelques exemples de radios (1 par classe) :")
    file_names = ["COVID-13.png", "Lung_Opacity-13.png", "Normal-13.png", "Viral Pneumonia-13.png"]
    cols = st.columns(4)
    for i in range(4):
        image_path = os.path.join(project_root, 'images', f"{file_names[i]}")
        with cols[i]:
            try:
                st.image(Image.open(image_path), caption=f"{file_names[i]}" ,use_container_width=True)
            except FileNotFoundError:
                st.markdown(f"_(Image non trouvée: `{file_names[i]}`)_")   


elif selection == "Analyse statistique":

    st.write("#### Distribution des classes")

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

    st.write("#### Distribution de la variance par classe")
        
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
    
    
    st.write("#### Normalisation des images")

    st.write("Il a été constaté que 7 radiographies sur 10 ne sont pas normalisées. Voici la représentation en fonction des diverses sources de données initiales :")
    
    input_filename = os.path.join(project_root, 'data', 'normalisation.csv')
    df_norm = pd.read_csv(input_filename)
    df_norm_sorted = df_norm.sort_values(by='is_norm', ascending=False)
    fig = px.bar(
        df_norm_sorted,
        y='url',
        x='percent',
        color='is_norm',
        title="Analyse de la normalisation des images selon la source de données"
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="% de radios normalisées",
        yaxis_title="Source",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("#### Identification des doublons")
    st.write("103 doublons ont été identifiés")

    input_filename = os.path.join(project_root, 'data', 'Doublons_liste.csv')
    df_doublons = pd.read_csv(input_filename)
    st.dataframe(df_doublons.rename(columns={'count': "Nombre de doublons", 'list': "Liste des doublons"}), hide_index=True)



st.subheader("Détection d'anomalies")

DESCRIPTIONS = {
    'IQR' : "Il s'agit de la méthode statistique classique basée sur l'Intervalle InterQuartile : les éléments situés hors de la plage Q1 - 1,5*IQR / Q3 + 1,5*IQR sont considérés comme des anomalies. Elle a été appliquée ici à l'intensité moyenne des pixels et à l'écart-type de l'internsité.",
    'Statistique': "Cette méthode fondamentale transforme chaque image en caractéristiques numériques (moyenne, contraste, entropie). Le score d'anomalie est basé sur la distance d'une image à la distribution normale. Idéal pour trouver des anomalies grossières comme des images très sombres ou vides.",
    'Isolation Forest': "Cette approche utilise un réseau expert (VGG16) pour extraire des caractéristiques complexes. L'algorithme Isolation Forest isole ensuite les images qui sont sémantiquement différentes des autres. Efficace pour trouver des textures ou des formes inhabituelles.",
    'Auto-encoder': "Un réseau de neurones est entraîné à compresser puis reconstruire les images du dataset. Il devient expert des radiographies 'typiques'. Une image qu'il peine à reconstruire (erreur élevée) est considérée comme anormale. C'est l'approche la plus sensible aux anomalies subtiles."
}

options = ["IQR", "Statistique", "Isolation Forest", "Auto-encoder"]
selection = st.segmented_control(
    "Choisissez la technique à visualiser",
    options,
    label_visibility="collapsed",
    default='IQR'
)

if selection == "IQR":
    st.write("#### Approche IQR")
    st.write(DESCRIPTIONS[selection])

    script_dir = os.path.dirname(os.path.abspath(__file__))    
    project_root = os.path.dirname(script_dir)    
    input_filename = os.path.join(project_root, 'data', 'iqr_outliers.csv')    
    df_plot = pd.read_csv(input_filename)
    nb_outliers = len(df_plot[df_plot['est_outlier'] == 'Oui'])

    st.write(f"""
    **{nb_outliers} outliers** ont été identifiés par cette méthode (calcul global sur tout le dataset).  
    Ci-dessous une visualisation de la répartition de l’intensité en fonction de l’écart-type sur les radios après normalisation :
    """)
   
    df_non_outliers = df_plot[df_plot['est_outlier'] == 'Non']
    df_outliers = df_plot[df_plot['est_outlier'] == 'Oui']

    fig = px.scatter(
        df_non_outliers,
        x='ecart_type',
        y='intensite_moyenne',
        color='classe',
        color_discrete_map=palette_bar,
        opacity=0.7, 
        category_orders={'classe': classes_order},
        labels={
            'intensite_moyenne': 'Intensité Moyenne (après normalisation)',
            'ecart_type': 'Écart-Type (Contraste/Texture)',
            'classe': 'Classe'
        }
    )

    fig.add_trace(
        go.Scatter(
            x=df_outliers['ecart_type'],
            y=df_outliers['intensite_moyenne'],
            mode='markers',
            marker=dict(
                symbol='x', 
                color='classe',
                color_discrete_map=palette_bar,
                opacity=1.0,
                line=dict(width=1.5)
            ),
            customdata=df_outliers[['classe']],
            name='Outlier' 
        )
    )

    fig.update_layout(
        legend_title="Légende",
        height=700,
        legend=dict(traceorder='normal') 
    )

    st.plotly_chart(fig, use_container_width=True)

    palette = {
    'Normal': 'green',
    'COVID': 'red',
    'Lung_Opacity': 'orange',
    'Viral Pneumonia': 'blue'}

    input_filename = os.path.join(project_root, 'data', 'intensity.csv')
    df_intensity = pd.read_csv(input_filename)   
    fig = px.scatter(
        df_intensity,
        x='norm_intensity_std',
        y='norm_intensity_mean',
        color='classification',
        title="Répartition intensité moyenne selon écart-type après normalisation",
        color_discrete_map=palette
    )
        
    fig.update_layout(
        xaxis_title="Ecart-type",
        yaxis_title="Intensité moyenne",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Exemples d'images anormales trouvées par la méthode IQR :")

    method_key = selection.lower().replace(' ', '_')  
    for i in range(3):
        cols = st.columns(5)
        for j in range(5):
            rank = i * 5 + j + 1
            script_dir = os.path.dirname(os.path.abspath(__file__))    
            project_root = os.path.dirname(script_dir)    
            image_path = os.path.join(project_root, 'outliers_images', f"{method_key}_{rank}.png")
            with cols[j]:
                try:
                    st.image(Image.open(image_path), use_container_width=True)
                except FileNotFoundError:
                    st.markdown(f"_(Image #{rank} non trouvée)_")    


elif selection == "Statistique":
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

