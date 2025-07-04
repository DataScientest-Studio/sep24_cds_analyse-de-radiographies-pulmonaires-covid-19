import streamlit as st
from utils import interactive_image

st.set_page_config(page_title="Interprétation", layout="wide")


st.title("Interprétation")

st.subheader("Analyse des erreurs")
st.markdown("""
La classe **Opacité pulmonaire** (non-COVID lung infection) pose de véritables difficultés pour la classification :

- Forte **variabilité visuelle** entre les images
- Recouvre parfois des cas proches comme **COVID** ou **pneumonie virale**
- Taux de **confusion élevé** avec les classes voisines (notamment *Normal* et *COVID*)

Elle constitue **la principale source d’erreurs** du modèle, comme le montre la matrice de confusion ci-dessous :
""")

#st.image("src/images/lung_opcaity_lenet.png")
st.image("src/images/lung_opacity_lenet.png")

st.subheader("Grad-CAM et interprétabilité")
st.write("""
Visualisation des zones activées par le modèle. Bonnes correspondances avec les zones pulmonaires atteintes.

L’image ci-dessous montre une carte Grad-CAM produite par EfficientNet pour une image classée comme COVID. 
On observe une activation marquée dans la région inférieure gauche du poumon — typique des atteintes liées à la COVID-19.

Attention toutefois : l’interprétation de ces cartes d’activation reste complexe dans un contexte médical.
""")

st.subheader("🧪 Exemple : activation sur une image *Normale*")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Image originale**")
    st.image("src/images/Normal-2.png", caption="Radiographie normale", use_container_width=True)

with col2:
    st.markdown("**Carte Grad-CAM**")
    st.image("src/images/gradcam_normal.png", caption="Grad-CAM - Image sans pathologie", use_container_width=True)

st.markdown("""
---

### 🧾 Lecture visuelle

- Le modèle **ne détecte pas de zones d’activation anormales** sur cette radiographie.
- La Grad-CAM est globalement **diffuse et peu marquée**, ce qui est **cohérent avec une image considérée comme saine**.
- Cela peut indiquer que le modèle ne surinterprète pas les images "normales", ce qui est un bon point en termes de **généralisation**.

""")

st.markdown("""

Ci-dessous, nous comparons trois variantes courantes :
- **Grad-CAM** : méthode de base
- **Layer-CAM** : plus localisée, utilise les gradients par couche
- **Grad-CAM++** : version affinée pour des objets multiples ou flous

""")

st.image("src/images/gradcam_comp.png", caption="Comparaison : GradCAM vs LayerCAM vs GradCAM++", use_column_width=True)

st.markdown("""
- Chaque ligne montre une image et ses activations selon la méthode utilisée.
- Les zones rouges indiquent les régions les plus contributives à la prédiction.
- On observe souvent une **concentration dans les bases pulmonaires ou les zones anormales** selon le cas.

> Ces outils ne remplacent pas une expertise médicale mais permettent **d'interpréter visuellement la décision du modèle**.

""")
