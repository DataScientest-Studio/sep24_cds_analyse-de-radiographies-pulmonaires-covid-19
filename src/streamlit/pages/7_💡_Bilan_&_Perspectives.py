import streamlit as st
from utils import interactive_image

st.set_page_config(page_title="Bilan et perspectives", layout="wide")

st.title("Bilan et perspectives")

st.subheader("Résultats obtenus")
st.write("""
Les modèles testés atteignent globalement d'excellents résultats (autour de 99 % de f1-score).
Le modèle **EfficientNetB0** a été retenu pour la démonstration finale en raison de son **excellent compromis entre performance et coût d'entraînement**.  
Il a atteint un **f1-score pondéré de 99.08 %**.
""")

st.subheader("Difficultés rencontrées")
st.write("""
- **Richesse et qualité du jeu de données** : 
  - Présence de matériel médical (électrodes d'ECG, cathéters, sondes, drains), 
  - Erreur d'acquisition et variabilité dans les techniques de réalisation : cadrage, flou / manque de contraste, position du patient, 
  - Annotations du radiologue (flèches, texte),
  - Représentativité de la population en terme de sexe et d'âge : sur-représentation de radiographies d'enfants pour la pneumonie par exemple.
- **Infrastructure et limites matérielles** : 
  - Echantillonage ciblé sur ~10 000 images (réprésentativité des différentes classes) pour le calcul des réductions de dimension pour ne pas saturer la mémoire,
  - Utilisation de GPU nécessaire pour l'entrainement, l'optimisation des hyperparamètes et le transfer-learning des CNN,
  - Réduction de la taille des batch pour éviter un crash mémoire.  
- **Diversité des pratiques et choix des paramètres** :
  - Taille des batchs : trop petite → mauvais apprentissage, trop grande → crash mémoire, compromis identifié autour de 32,
  - Prétraitement des images : nombreuses techniques dans la littérature pour éliminer les anomalies, améliorer le contraste, éliminer le bruit, etc.
- **Temps limité** : projet mené en parallèle de la formation et des obligations professionnelles.
""")

st.subheader("Perspectives futures")
st.write("""
- **Augmentation du jeu de données** : apport des sources de données cliniques supplémentaires pour garantir la généralisation et limiter le biais d'apprentissage,
- **Intégration de métadonnées cliniques** (si disponibles) : âge, sexe, antécédents, symptômes, résultats biologiques, etc.,
- **Segmentation préalable des poumons** : segmentation automatique des poumons (par exemple avec U-Net) avant la classification pour concentrer l’attention du modèle sur les régions d’intérêt et réduire l’influence du bruit hors poumon,
- **Extraction de textures avancées** : ajout de descripteurs texturaux (par exemple GLCM, GLDM ou ondelettes) pour enrichir les features et mieux différencier les pathologies similaires,
- **Mise en place d'optimisateurs avancés** : algorithmes (par exemple Adamax) pour optimiser la convergence et la stabilité du modèle,
- **Interprétabilité avancée** : utilisation des cartes d'attentions de modèles hybrides CNN & Vision Transformer, techniques récentes (SHAP, etc.),
- **Mise en oeuvre d'architectures avancées** : 
  - Approches d'ensemble : plusieurs architectures CNN entraînées séparément dont dont les prédictions sont aggrégées (moyenne ou vote majoritaire),
  - Architectures hybrides (CNN & ViT) : qui capturent à la fois les dépendances locales et globales et qui apportent d'autres éléments d'interprétabilité,
""")
with st.expander("🔬 DenseNet-121 + Vision Transformer (ViT)"):
    st.write("""
    DenseNet-121 est un réseau CNN structuré en quatre blocs denses. Chaque couche reçoit les sorties de toutes les couches précédentes, améliorant ainsi la réutilisation des caractéristiques extraites et limitant la disparition du gradient.  
    ViT (Vision Transformer), quant à lui, segmente l’image en patches pour appliquer une self-attention globale.  
    Cette approche hybride exploite les forces combinées :  
    - du **CNN** → qui capture les dépendances **locales** 
    - du **Vision Transformer** → qui capture les dépendances **globales** 

    **f1-score pondéré : 0.99, précision pondérée : 0.99**  
    Optimisé via Grid Search sur le learning rate, le taux de dropout, la taille des batchs
    """)
st.write("""
  - Intégration de modules d’attention : pour améliorer la focalisation du réseau sur les régions critiques.
""")
with st.expander("🔬 VGG16 + SE (Squeeze-and-Excitation)"):
    st.write("""
    VGG16 est un CNN profond à 16 couches, introduit en 2014.  
    Les blocs **SE (Squeeze-and-Excitation)** appliquent une **attention par canal**, permettant au réseau de **recalibrer dynamiquement** les canaux d’activation.  
    Cette attention est **insérée après chaque bloc convolutif**, avant le max-pooling.
    **f1-score pondéré : 0.9864, précision pondérée : 0.9864**
    """)
st.write(""" 
  → Contrairement aux attentes, ces modèles avancés ne surpassent pas significativement les modèles CNN fine-tunés comme EfficientNet ou InceptionV3.  
En revanche, leur coût en calcul est plus élevé, notamment à cause des modules d’attention.
""")

st.subheader("Conclusion")
st.write("""
Ce projet a permis de mettre en œuvre une **démarche complète d’analyse et de classification d’images médicales**, depuis le prétraitement des radiographies pulmonaires jusqu’à l’évaluation de modèles avancés de deep learning. Après avoir exploré les limites des approches classiques de machine learning, nous avons démontré la **pertinence et l’efficacité des réseaux de neurones convolutifs**.
\n\nL’optimisation des hyperparamètres, le fine-tuning du modèle, ainsi que l’application de techniques de prétraitement et d’augmentation de données ont permis d’atteindre une performance satisfaisante, validant la capacité du modèle à extraire des caractéristiques discriminantes au sein d’images médicales complexes.
L’intégration de méthodes d’explicabilité, telles que Grad-CAM et ses variantes, ont mis en évidence les régions d’intérêt pertinentes pour la décision.
\n\nPlusieurs perspectives d’amélioration subsistent, notamment l’exploration de modèles hybrides, l’utilisation de techniques avancées de prétraitement ainsi que l'augmentation de jeu de données.
\n\nAu-delà des résultats obtenus, ce projet souligne le **potentiel de l’intelligence artificielle pour assister les professionnels de santé dans le diagnostic par imagerie, tout en mettant en lumière les défis liés à la qualité des données, à l’équilibrage des classes et à l’interprétabilité des modèles**.
""")
