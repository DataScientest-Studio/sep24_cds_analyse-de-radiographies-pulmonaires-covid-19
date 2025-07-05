import streamlit as st
from utils import interactive_image

st.set_page_config(page_title="Bilan et perspectives", layout="wide")


st.title("Bilan et perspectives")

st.subheader("Résultats obtenus")
st.write("""
Les modèles testés atteignent globalement d'excellents résultats (autour de 99 % de F1-score).
Le modèle **EfficientNetB0** a été retenu pour la démo finale en raison de son **excellent compromis entre performance et coût d'entraînement**.  
Il a atteint un **F1-score pondéré de 99.08 %**.
""")

st.subheader("Difficultés rencontrées")
st.write("""
- **Données non homogènes** : forte variabilité des images selon leur origine.
- **Classes déséquilibrées** : certains labels sont sous-représentés, rendant l'apprentissage plus difficile.
- **Infrastructure GPU nécessaire** : les entraînements avancés nécessitent des ressources que nos PC personnels ne permettaient pas.  
  L’usage des GPU gratuits de **Kaggle (30h/semaine)** a été essentiel pour tester le dégel des couches convolutives.
- **Gestion de la mémoire GPU** : lors du dégel de plusieurs blocs convolutifs (ex. : VGG16 en 4 classes), il a été nécessaire de réduire la taille des batchs (batch = 32).
- **Choix de la taille des batchs** :  
  - Trop petite : mauvais apprentissage (~20 % d’accuracy en VGG16 3 classes avec batch=32).  
  - Trop grande : overfitting ou crash mémoire.  
  - **Compromis idéal identifié autour de batch=32**, selon les résultats de la recherche d’hyperparamètres.
- **Prétraitement des images** : divergence des méthodes selon les cas (élimination des anomalies, contraste, bruit…).
- **Temps limité** : projet mené en parallèle de la formation et des obligations professionnelles.
""")

st.subheader("Perspectives futures")
st.write("""
- **Intégration de métadonnées cliniques** : âge, sexe, antécédents, symptômes…
- **Exploration de nouvelles architectures** :  
  Ex. : **Gravitational Search Algorithm** pour optimiser les hyperparamètres.
- **Amélioration de l’interprétabilité** :  
  Le modèle **DenseNet-121 + Vision Transformer** permet une meilleure compréhension via des **attention maps** et **Grad-CAM++**.
""")

st.set_page_config(page_title="Modèles avancés", layout="wide")



st.subheader("Modèles Avancés")
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
