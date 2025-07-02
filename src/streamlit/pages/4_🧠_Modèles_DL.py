import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import interactive_image
import plotly.express as px
from codecarbon import EmissionsTracker

st.set_page_config(layout="wide")
st.title("📊 Résultats des modèles deep learning")

# Méthodologie
st.header("🔧 Méthodologie")
st.markdown("""
- **Transfer learning + fine‑tuning** sur les dernières couches des modèles pré-entraînés ImageNet  
- **Suppression de la classe** 'opacité pulmonaire' → uniquement **Covid**, **Normal**, **Pneumonie virale**
""")

# Données enrichies avec années
data = [
    {"Année": 1998, "Modèle": "LeNet", "Params totaux": 61111, "Params fine‑tuning": 61111, "Temps/epoch (s)": 25, "Précision (%)": 91.36, "Rappel (%)": 90.60, "F1-score (%)": 90.78},
    {"Année": 2014, "Modèle": "Inception", "Params totaux": 22328099, "Params fine‑tuning": 22293667, "Temps/epoch (s)": 76, "Précision (%)": 98.09, "Rappel (%)": 97.36, "F1-score (%)": 98.55},
    {"Année": 2015, "Modèle": "ResNet", "Params totaux": 29886340, "Params fine‑tuning": 6298628, "Temps/epoch (s)": 150, "Précision (%)": 99.30, "Rappel (%)": 98.85, "F1-score (%)": 99.08},
    {"Année": 2019, "Modèle": "EfficientNetB0", "Params totaux": 5701286, "Params fine‑tuning": 5656703, "Temps/epoch (s)": 66, "Précision (%)": 99.08, "Rappel (%)": 99.08, "F1-score (%)": 99.08},
    {"Année": 2016, "Modèle": "DenseNet-121", "Params totaux": 6956931, "Params fine‑tuning": 4588035, "Temps/epoch (s)": 115, "Précision (%)": 98.49, "Rappel (%)": 98.48, "F1-score (%)": 98.48},
    {"Année": 2014, "Modèle": "VGG16", "Params totaux": 134272835, "Params fine‑tuning": 126637571, "Temps/epoch (s)": 100, "Précision (%)": 99.31, "Rappel (%)": 99.31, "F1-score (%)": 99.31},
]

# Création du DataFrame trié
df = pd.DataFrame(data).sort_values("Année")

st.header("📋 Performances par modèle (3 classes)")
st.dataframe(df.style.format({
    "Params totaux": "{:,.0f}",
    "Params fine‑tuning": "{:,.0f}",
    "Temps/epoch (s)": "{:.0f}",
    "Précision (%)": "{:.2f}",
    "Rappel (%)": "{:.2f}",
    "F1-score (%)": "{:.2f}",
}), hide_index=True)

# Visualisations

st.header("📈 Comparaisons visuelles")

# Transformation des colonnes pour line plot
melted = df.melt(id_vars="Modèle", value_vars=["Précision (%)", "Rappel (%)", "F1-score (%)"],
                 var_name="Métrique", value_name="Valeur")

fig2 = px.line(
    melted,
    x="Modèle",
    y="Valeur",
    color="Métrique",
    markers=True,
    labels={"Valeur": "Score (%)"}
)

fig2.update_layout(title="Comparaison des scores (Précision, Rappel, F1-score)", title_x=0.3)
st.plotly_chart(fig2, use_container_width=True)

fig1 = px.scatter(
    df,
    x="Params totaux",
    y="Temps/epoch (s)",
    size="F1-score (%)",
    color="Modèle",
    hover_name="Modèle",
    labels={
        "Params totaux": "Paramètres (totaux)",
        "Temps/epoch (s)": "Temps/époque (s)",
        "F1-score (%)": "F1-score (%)"
    }
)

# Centrage du titre
fig1.update_layout(title="Temps d'entraînement vs Taille du modèle", title_x=0.3)
st.plotly_chart(fig1, use_container_width=True)


# Focus sur EfficientNet
st.header("⭐ Focus sur **EfficientNetB0**")
eff = df[df.Modèle=="EfficientNetB0"].iloc[0]
st.markdown(f"""
- **Année** : {eff["Année"]} → modèle récent et optimisé  
- **Params totaux** : {eff["Params totaux"]:,} (~5.7 M)  
- **Params fine‑tuning** : {eff["Params fine‑tuning"]:,} (~99 % des paramètres)  
- **Temps/epoch** : {eff["Temps/epoch (s)"]} s — deux fois plus rapide que ResNet et VGG  
- **F1‑score** : {eff["F1-score (%)"]:.2f} % → ↑ haute performance tout en restant léger

EfficientNetB0 incarne le compromis idéal **sobriété vs performance**, permettant d'obtenir d'excellents résultats (≈ 99 %) avec un modèle compact et rapide, idéal pour le déploiement.
""")

st.markdown("""
**✅ Conclusion :**
- Tous les modèles surpassent 98 % de F1‑score, EfficientNetB0 se distingue par sa compacité et son efficacité.
- Utile pour les déploiements contraints en ressources (edge, cloud limité, etc.).
""")


st.markdown("---")
st.subheader("🧪 Essai avec une radiographie")
uploaded_file = st.file_uploader("Chargez une radiographie", type=["jpg", "jpeg", "png"])

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
    st.image(image, caption="Image chargée")

    # Initialisation du tracker
    tracker = EmissionsTracker(project_name="streamlit_inference")
    tracker.start()

    with st.spinner("Prédiction en cours..."):
        input_tensor = preprocess_image(image)
        predictions = model.predict(input_tensor)[0]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)

    st.markdown(f"**Classe prédite :** `{predicted_class}`")
    st.markdown(f"**Confiance :** `{confidence:.2f}%`")
    st.bar_chart(dict(zip(class_names, predictions)))

    # Arrêt du tracker et affichage des émissions
    tracker.stop()
    st.write(f"Émissions estimées lors de l'inférence : {tracker.final_emissions*1000:.2e} g CO₂ (Estimation Code Carbone)")
