import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import interactive_image


st.set_page_config(page_title="Modèles de Deep Learning", layout="wide")


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
st.image("src/images/DeepSynthese.png", caption="Synthèse des performances des modèles CNN", width=750)

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
    st.image("src/images/umap_sans.png", caption="Représentation UMAP sans la classe d’opacité", width=700)

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

st.markdown("---")
st.subheader("🧪 Essai avec une radiographie")
uploaded_file = st.file_uploader("Téléversez une radiographie", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/efficientnet_final.h5")

model = load_model()
class_names = ["COVID", "Normal", "Viral Pneumonia"]

def preprocess_image(image):
    image = image.convert("RGB").resize((240, 240))
    return np.expand_dims(np.array(image) / 255.0, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléversée", use_container_width=True)

    with st.spinner("Prédiction en cours..."):
        input_tensor = preprocess_image(image)
        predictions = model.predict(input_tensor)[0]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)

    st.markdown(f"**Classe prédite :** `{predicted_class}`")
    st.markdown(f"**Confiance :** `{confidence:.2f}%`")
    st.bar_chart(dict(zip(class_names, predictions)))
