import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd  
import plotly.express as px
import plotly.graph_objects as go
import os


st.write("Répertoire courant :", os.getcwd())
st.write("Contenu du répertoire :", os.listdir())
st.write("Contenu de ../images :", os.listdir("../images") if os.path.exists("../images") else "Dossier 'images' non trouvé")

# Configuration de la page
st.set_page_config(page_title="Analyse COVID-19 Radiographies", layout="wide")

col1, col2 = st.columns([1, 10])  # Puedes ajustar la proporción
with col1:
    st.image("../images/Normal-2.png", width=170)
with col2:
    st.title("Détection de pathologies pulmonaires à partir de radiographies thoraciques")

st.markdown("**Projet basé sur le rapport d'analyse de radiographies pour la détection de la COVID-19 et autres pathologies.**")

# Menu de navigation
section = st.sidebar.radio("\U0001F4DA Navigation", [
    "1. Introduction",
    "2. Analyse des données",
    "3. Méthodologie",
    "4. Modelisation",
    "5. Modèles de Deep Learning",
    "6. Modèles Avancés",
    "7. Résultats et Interprétation",
    "8. Bilan et perspectives",
    "9. Essai avec une radiographie"
])

def interactive_image(path, caption=None):
    img = Image.open(path)
    width, height = img.size
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=img,
            x=0, y=height,
            sizex=width, sizey=height,
            xref="x", yref="y",
            sizing="contain",
            layer="below"
        )
    )

    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height])
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="zoom",
        title=caption if caption else ""
    )

    st.plotly_chart(fig, use_container_width=True)

# Sections
if section == "1. Introduction":
    st.title("Introduction")
    st.write("""
    Le diagnostic médical par imagerie radiologique est un élément clé de la médecine moderne. Toutefois, l’interprétation manuelle des radiographies thoraciques est souvent complexe, sujette à erreurs humaines et dépend fortement de l’expertise du radiologue.

    Ce projet, réalisé dans le cadre de la formation Data Scientist - Septembre 2024 chez Data Scientest, vise à développer un pipeline automatique de classification des radiographies pulmonaires. L’objectif principal est la détection précise de la COVID-19, de la pneumonie virale et d'autres pathologies pulmonaires.

    La méthodologie comprend une exploration et un pré-traitement approfondis des données, suivi de la comparaison entre modèles classiques de Machine Learning et architectures avancées de Deep Learning, incluant des techniques de transfer learning et des modèles hybrides CNN-ViT.

    Les performances des modèles sont évaluées via plusieurs métriques clés : précision, sensibilité (rappel), spécificité, F1-score pondéré et matrice de confusion.
    """)


elif section == "2. Analyse des données":
    st.title("Analyse des données")
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
    interactive_image("../images/DistributionClasses.png", "exemple")

    st.write("""
    Visualisation statistique : variance de l’intensité, projections UMAP, et examen manuel sur quelques images.
    Variance : ci-dessous une visualisation de la variance par classe
    """)
    interactive_image("../images/Variance.png", "exemple")

    st.write("Intensité vs. écart-type : ci-dessous une visualisation de la répartition de l’intensité en fonction de l’écart-type sur les radios après normalisation :")
    interactive_image("../images/Intensite-ecart.png", "exemple")

    st.write("""
    Projection 2D après réduction de dimension via PCA (linéaire) et normalisation préalable :
    """)
    interactive_image("../images/Projection2d.png", "exemple")

    st.write("""
    Projection 2D après réduction de dimension via UMAP non linéaire (High Performance Dimension Reduction) et normalisation préalable:
    """)
    interactive_image("../images/UMAP.png", "exemple")

    st.write("""
    Projection 2D après encodage / décodage par auto-encodeur (AE) et: normalisation préalable
    """)
    interactive_image("../images/Autoencoder.png", "exemple")

    st.write("""
    Inspection visuelle de quelques images : l’inspection visuelle met en évidence que les radios sont dans l’ensemble de très bonne qualité.
    """)
    interactive_image("../images/InspectionVisuelle.png", "exemple")

    st.subheader("Prétraitement")
    st.write("""
    Les images ont été redimensionnées à 240x240 pixels, normalisées, et enrichies par augmentation de données (flip, rotation, zoom). Des méthodes comme Isolation Forest ont été utilisées pour retirer les outliers.

    Il a été constaté que 7 radiographies sur 10 ne sont pas normalisées. Voici la représentation en fonction des diverses sources de données initiales :            
    """)
    interactive_image("../images/Normalisation.png", "exemple")
    

elif section == "3. Méthodologie":
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


elif section == "4. Modelisation":
    st.title("Modèles de Machine Learning")
    st.header("Résultats des modèles classiques")

    model_info = {
        "KNN": {
            "description": """KNN est un algorithme d’apprentissage supervisé introduit dans les années 1950. Il prédit la classe d’un échantillon en se basant sur les **K voisins les plus proches** dans l’espace des caractéristiques, en utilisant une métrique de distance comme la distance euclidienne.""",
            "metrics": "**F1-score : 77 %, Accuracy : 83 %**"
        },
        "Random Forest": {
            "description": """Random Forest, proposé par Leo Breiman en 2001, est un ensemble d’**arbres de décision** entraînés sur des sous-échantillons aléatoires des données (méthode bootstrap). Il utilise également une sélection aléatoire de variables pour réduire la corrélation entre les arbres, ce qui permet de **réduire la variance globale** du modèle.""",
            "metrics": "**F1-score : 83 %, Accuracy : 86 %**"
        },
        "SVM": {
            "description": """SVM est un algorithme de classification supervisée qui cherche à **maximiser la marge** entre les classes en trouvant l’hyperplan optimal. Il est particulièrement efficace pour les problèmes linéairement séparables ou faiblement bruités.""",
            "metrics": "**F1-score : 82 %, Accuracy : 85 %**"
        },
        "XGBoost": {
            "description": """XGBoost est une méthode d’ensemble basée sur le **gradient boosting**, introduite par Tianqi Chen en 2016. Il construit une séquence de modèles faibles (arbres peu profonds) où chaque modèle suivant corrige les erreurs du précédent. Il est reconnu pour son **efficacité et performance en compétition**.""",
            "metrics": "**F1-score : 86 %, Accuracy : 88 %**"
        },
        "MLPClassifier": {
            "description": """MLPClassifier est un **réseau de neurones artificiels** à propagation avant (feedforward) composé de plusieurs couches : entrée, cachée(s) et sortie. Chaque neurone applique une fonction d’activation non linéaire, et l’apprentissage est réalisé par **rétropropagation du gradient** (souvent avec Adam ou SGD).""",
            "metrics": "**F1-score : 81 %, Accuracy : 84 %**"
        }
    }

    selected_model = st.selectbox("Sélectionnez un modèle pour afficher les détails :", list(model_info.keys()))

    st.subheader(selected_model)
    st.write(model_info[selected_model]["description"])
    st.markdown(model_info[selected_model]["metrics"])
    st.markdown("---")

    st.write("Conclusion et tableau de synthèse")
    st.image("../images/TablaModelos.png", caption="exemple", width=750)

    # Continúa con el resto de las subsecciones como estaban
    st.title("Optimisation des modèles ML")

    st.subheader("🔍 Grid Search")
    st.write("""Grid Search est une méthode d’optimisation des hyperparamètres...""")

    st.subheader("📈 HOG (Histogram of Oriented Gradients)")
    st.write("""Le descripteur HOG, introduit par Dalal et Triggs en 2005...""")

    st.subheader("⚖️ Standardisation des données")
    st.write("""La standardisation met les variables sur des échelles comparables...""")

    st.subheader("🖼️ Effet de la taille des images")
    st.write("""Une image plus grande contient davantage d’information visuelle...""")

    st.subheader("📊 Panel de données et impact du sampling")
    st.write("### Undersampling")
    st.write("""Une réduction aléatoire de la taille du jeu d’entraînement montre que...""")

    st.write("### Oversampling avec SMOTE")
    st.write("""La technique **SMOTE** permet de générer artificiellement...""")



elif section == "5. Modèles de Deep Learning":
    st.title("Modèles de Deep Learning")

    st.write("""
Les modèles **CNN préentraînés fine-tunés** ont largement surpassé les modèles classiques de machine learning, grâce à leur capacité à capturer des caractéristiques complexes dans les images médicales.
""")

    st.markdown("---")
    st.write("## 🧠 Modèles explorés")

    with st.expander("📌 VGG16"):
        st.write("""
Développé par l’équipe du Visual Geometry Group (VGG) à l’Université d’Oxford, VGG16 a été proposé en 2014 et a marqué une avancée majeure dans la vision par ordinateur. Son architecture simple et profonde repose sur des **convolutions 3x3 empilées**.  
**F1-score : 99.31 %, Accuracy : 99.31 %**
""")

    with st.expander("📌 InceptionV3"):
        st.write("""
Modèle introduit par Google en 2015, InceptionV3 améliore les versions précédentes d’Inception/GoogLeNet. Il utilise des blocs "Inception" composés de **convolutions de différentes tailles**, ce qui permet de capter plusieurs échelles d'information simultanément.  
**F1-score : 99.02 %, Accuracy : 99.02 %**
""")

    with st.expander("📌 LeNet-5"):
        st.write("""
L’un des tout premiers CNN opérationnels, proposé par Yann LeCun en 1998. Utilisé initialement pour la reconnaissance de chiffres manuscrits (MNIST), LeNet est un modèle simple mais historique, ayant posé les bases du deep learning moderne.  
**F1-score : 91 %, Accuracy : 93 %**
""")

    with st.expander("📌 ResNet"):
        st.write("""
Proposé en 2015 par Kaiming He (Microsoft Research), ResNet introduit les **connexions résiduelles**, qui permettent d’entraîner des réseaux très profonds sans perte de performance. Cette innovation a révolutionné l'apprentissage profond.  
**F1-score : 99.19 %, Accuracy : 99.19 %**
""")

    with st.expander("📌 EfficientNetB0"):
        st.write("""
Présenté par Google Brain en 2019, EfficientNet introduit un **scaling uniforme** des dimensions (profondeur, largeur, résolution) du réseau. Il atteint une **meilleure efficacité et précision** avec un nombre de paramètres réduit.  
**F1-score : 99.08 %, Accuracy : 99.08 %**
""")

    with st.expander("📌 DenseNet-121"):
        st.write("""
Proposé en 2017 par Gao Huang, DenseNet se distingue par sa **connectivité dense entre les couches**. Chaque couche reçoit comme entrée les sorties de toutes les couches précédentes dans le bloc. Cette stratégie favorise une meilleure réutilisation des caractéristiques extraites.  
**F1-score : 99.04 %, Accuracy : 99.04 %**
""")

    st.write("""
### ✅ Conclusion et tableau de synthèse
Nette amélioration par rapport aux modèles de machine learning classiques : **F1-score global > 98 %** pour la classification 3 classes (hors LeNet qui est à 90 %).
""")
    st.image("../images/DeepSynthese.png", caption="Synthèse des performances des modèles CNN", width=750)

    st.markdown("---")
    st.subheader("🔧 Optimisation des modèles deep learning")

    with st.expander("📐 Effet de la taille des images"):
        st.write("""
Le graphique ci-dessus illustre l’évolution de la précision et de la loss pour différentes tailles d’images (32×32, 64×64, 128×128, 240×240), en fonction du nombre d’époques.  
On observe un gain notable en précision de validation, passant de **~80 %** avec des images 32×32 à **plus de 90 %** avec des images 240×240.  
Contrairement aux modèles classiques, les CNN bénéficient d’images en haute résolution.  
**➡️ Les images 240×240 offrent le meilleur compromis performance/précision.**
""")

    with st.expander("🚫 Impact de la classe d’opacité pulmonaire"):
        st.write("""
Dans la classification 4 classes, la classe d’opacité pulmonaire n’est correctement prédite que dans **82 %** des cas, bien en dessous des autres.  
Elle regroupe des pathologies non-COVID très diverses et peu homogènes.  
En retirant cette classe, la classification (3 classes) gagne en précision (souvent >95 %).  
**➡️ Décision : retirer la classe d’opacité pulmonaire pour améliorer la clarté du modèle.**
""")
        st.image("../images/umap_sans.png", caption="Représentation UMAP sans la classe d’opacité", width=700)

    with st.expander("🔍 Optimisation des hyperparamètres avec Optuna / Keras Tuner"):
        st.write("""
L’optimisation des hyperparamètres sur EfficientNet a permis un gain significatif de performance :
- 📈 Scores par classe jusqu’à **99 %** (contre 95 % sans tuning)
- 🧪 Tuning effectué sur :  
  - le **learning rate**  
  - la **taille des couches denses**  
  - le **dropout**

➡️ L'impact est particulièrement visible dans les matrices de confusion après tuning.
""")

    with st.expander("😷 Effet des masques sur les performances"):
        st.write("""
Test effectué avec LeNet sur deux jeux de données : avec et sans masques.  
Résultat :  
- Les masques entraînent une **dégradation systématique** des performances (Précision, Rappel, F1).  
- Cela pourrait s’expliquer par la **perte d’informations clés** dans la zone du visage ou du thorax.

**➡️ Conclusion : l’usage des masques, dans ce cas, n’est pas bénéfique pour l'entraînement.**
""")




elif section == "6. Modèles Avancés":
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


elif section == "7. Résultats et Interprétation":
    st.title("Résultats et Interprétation")
    
    st.subheader("Analyse des erreurs")
    st.write("""
    La classe 'Opacité pulmonaire' introduit beaucoup de confusion. Les erreurs sont rares sur 'COVID' et 'Normal'.
    """)

    st.subheader("Grad-CAM et interprétabilité")
    st.write("""
    Visualisation des zones activées par le modèle. Bonnes correspondances avec les zones pulmonaires atteintes.
    
    L’image ci-dessous montre une carte Grad-CAM produite par EfficientNet pour une image classée comme COVID. 
    On observe une activation marquée dans la région inférieure gauche du poumon — typique des atteintes liées à la COVID-19.
    
    Attention toutefois : l’interprétation de ces cartes d’activation reste complexe dans un contexte médical.
    """)

    st.image("images/gradcam_covid.png", caption="Activation Grad-CAM sur une image classée comme COVID", width=400)

elif section == "8. Bilan et perspectives":
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


elif section == "9. Essai avec une radiographie":
    st.title("🧪 Essai avec une radiographie")
    uploaded_file = st.file_uploader("Téléversez une radiographie", type=["jpg", "jpeg", "png"])

    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("efficientnet_final.h5")

    model = load_model()
    class_names = ["COVID", "Normal", "Viral Pneumonia"]

    def preprocess_image(image):
        image = image.convert("RGB").resize((240, 240))
        return np.expand_dims(np.array(image) / 255.0, axis=0)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléversée", use_column_width=True)

        with st.spinner("Prédiction en cours..."):
            input_tensor = preprocess_image(image)
            predictions = model.predict(input_tensor)[0]
            predicted_class = class_names[np.argmax(predictions)]
            confidence = 100 * np.max(predictions)

        st.markdown(f"**Classe prédite :** `{predicted_class}`")
        st.markdown(f"**Confiance :** `{confidence:.2f}%`")
        st.bar_chart(dict(zip(class_names, predictions)))
