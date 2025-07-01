import streamlit as st
from utils import interactive_image


st.set_page_config(page_title="Modèles avancés", layout="wide")



st.title("Modèles Avancés")
st.write("""
Les modèles avancés explorent la combinaison de **CNN classiques avec des modules d’attention** ou des architectures de type **Transformer**.
""")
st.markdown("---")
st.subheader("🔬 Architectures hybrides explorées")

with st.expander("🧠 DenseNet-121 + Vision Transformer (ViT)"):
    st.write("""
    DenseNet-121 est un réseau CNN structuré en quatre blocs denses. Chaque couche reçoit les sorties de toutes les couches précédentes, améliorant ainsi la réutilisation des caractéristiques extraites et limitant la disparition du gradient.  
    ViT (Vision Transformer), quant à lui, segmente l’image en **patches** pour appliquer une **self-attention globale**.  
    Cette approche hybride exploite **les forces combinées** :  
    - locales → CNN  
    - globales → Transformer  

    **F1-score pondéré : 0.99, Précision pondérée : 0.99**  
    Optimisé via **Grid Search** sur :  
    - le learning rate  
    - le taux de dropout  
    - la taille des batchs
    """)

with st.expander("📦 VGG16 + SE (Squeeze-and-Excitation)"):
    st.write("""
    VGG16 est un CNN profond à 16 couches, introduit en 2014.  
    Les blocs SE (Squeeze-and-Excitation) appliquent une **attention par canal**, permettant au réseau de **recalibrer dynamiquement** les canaux d’activation.  
    Cette attention est insérée après chaque bloc convolutif, avant le max-pooling.

    **F1-score pondéré : 0.9864, Précision pondérée : 0.9864**
    """)

st.markdown("---")
st.subheader("📉 Conclusion")
st.write("""
Contrairement aux attentes, ces modèles avancés **ne surpassent pas significativement** les modèles CNN fine-tunés comme EfficientNet ou InceptionV3.  
En revanche, leur coût en calcul est **nettement plus élevé**, notamment à cause des modules d’attention.
""")