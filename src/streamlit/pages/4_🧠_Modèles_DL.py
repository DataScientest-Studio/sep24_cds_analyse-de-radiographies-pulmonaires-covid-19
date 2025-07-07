import os
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import interactive_image
import plotly.express as px
from codecarbon import EmissionsTracker
import gdown
from tensorflow.keras.applications.efficientnet import preprocess_input
import torch
from torchvision import transforms
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn



st.set_page_config(layout="wide")
st.title("📊 Résultats des modèles deep learning")

# Méthodologie
st.header("🔧 Méthodologie")
st.markdown("""
- **Test d'un CNN de référence (LeNet)** puis de plusieurs modèles représentatifs d'évolutions successives.  
Pour chacun de ces modèles, la logique suivante a été appliquée :  
- **Transfer learning** sur la base de modèles pré-entraînés ImageNet
- **Optimisation d'hyper-paramètres** par keras-tuner ou optuna (couches de classification en particulier : nb de couches/neurones)
- **Fine‑tuning** via dégel des dernières couches de convolution des modèles pré-entraînés
- **Suppression de la classe 'Opacité pulmonaire'** → uniquement Covid, Normal, Pneumonie virale.  
        La classe d’opacité pulmonaire est définie comme regroupant des cas d’infections pulmonaires non liées au COVID-19, une définition large et peu spécifique.
        Il est probable qu’elle contienne un mélange hétérogène de pathologies pulmonaires. Sa détection était moins bonne, avec des résultats globaux de 4% à 5% inférieurs.
""")

# Données enrichies avec années
data = [
    {"Année": 1998, "Modèle": "LeNet", "Params totaux": 61111, "Params fine‑tuning": 61111, "Temps/epoch (s)": 25, "Précision (%)": 91.36, "Rappel (%)": 90.60, "F1-score (%)": 90.78},
    {"Année": 2015, "Modèle": "Inception", "Params totaux": 22328099, "Params fine‑tuning": 22293667, "Temps/epoch (s)": 76, "Précision (%)": 98.55, "Rappel (%)": 98.54, "F1-score (%)": 98.55},
    {"Année": 2015, "Modèle": "ResNet", "Params totaux": 29886340, "Params fine‑tuning": 6298628, "Temps/epoch (s)": 150, "Précision (%)": 99.30, "Rappel (%)": 98.85, "F1-score (%)": 99.08},
    {"Année": 2019, "Modèle": "EfficientNetB0", "Params totaux": 5701286, "Params fine‑tuning": 5656703, "Temps/epoch (s)": 66, "Précision (%)": 99.08, "Rappel (%)": 99.08, "F1-score (%)": 99.08},
    {"Année": 2017, "Modèle": "DenseNet-121", "Params totaux": 6956931, "Params fine‑tuning": 4588035, "Temps/epoch (s)": 115, "Précision (%)": 98.49, "Rappel (%)": 98.48, "F1-score (%)": 98.48},
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

# Création du graphique à barres
fig2 = px.bar(
    melted,
    x="Modèle",
    y="Valeur",
    color="Métrique",
    barmode="group",  # Affiche les barres côte à côte
    labels={"Valeur": "Score (%)"}
)

fig2.update_layout(
    title="Comparaison des scores (Précision, Rappel, F1-score)",
    title_x=0.3,
    bargap=0.3,         # Espace entre les groupes de barres
    bargroupgap=0.15    # Espace entre les barres dans un groupe
)

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
- **Temps/epoch** : {eff["Temps/epoch (s)"]} s — deux fois plus rapide que ResNet et VGG  
- **F1‑score** : {eff["F1-score (%)"]:.2f} % → ↑ haute performance tout en restant léger

EfficientNetB0 incarne le compromis idéal **sobriété vs performance**, permettant d'obtenir d'excellents résultats (≈ 99 %) avec un modèle compact et rapide, idéal pour le déploiement.
""")

st.markdown("""
**✅ Conclusion :**
- Tous les modèles surpassent 98 % de F1‑score, EfficientNetB0 se distingue par sa compacité et son efficacité.
- Utile pour les déploiements contraints en ressources (cloud limité, mobilité...).
""")


st.markdown("---")
st.subheader("🧪 Essai avec une radiographie")
uploaded_file = st.file_uploader("Chargez une radiographie", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("src/models/efficientnet_optimized.h5")

#model = load_model()

#class_names = ["COVID", "Normal", "Viral Pneumonia"]

def preprocess_image(image_pil):
    image_resized = image_pil.convert("RGB").resize((240, 240))
    img_array = np.array(image_resized) 
    img_array_preprocessed = preprocess_input(img_array)
    input_tensor = np.expand_dims(img_array_preprocessed, axis=0)    
    return input_tensor

def preprocess_image(image):
    image = image.convert("RGB").resize((240, 240))
    img_array = np.array(image)
    img_array = preprocess_input(img_array)  
    return np.expand_dims(img_array, axis=0)


def predict_image(image_path, model, class_names, device="cpu"):
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = image_path.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probs).item()
        predicted_class = class_names[predicted_idx]
        confidence = probs[predicted_idx].item() * 100

    #print(f"Prédiction : {predicted_class} ({confidence*100:.2f}%)")
    return predicted_class, confidence, probs

class EfficientNetClassifierOptimized(nn.Module):
    def __init__(self, num_classes=4, fine_tune=False):
        super(EfficientNetClassifierOptimized, self).__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b0(weights=weights)

        for param in self.base_model.parameters():
            param.requires_grad = fine_tune

        in_features = self.base_model.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 3072),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3072, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes)
        )

        self.base_model.classifier = nn.Identity()

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x
    
model = EfficientNetClassifierOptimized(num_classes=4)
model.load_state_dict(torch.load("models/efficentnetB0.pth", map_location="cpu"))
model.eval()    

class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée")

    # Initialisation du tracker
    tracker = EmissionsTracker(project_name="streamlit_inference")
    tracker.start()

    with st.spinner("Prédiction en cours..."):
        #input_tensor = preprocess_image(image)
        #predictions = model.predict(input_tensor)[0]
        predicted_class, confidence, predictions =  predict_image(image, model, class_names)
        #st.markdown(predictions)
        #predicted_class = class_names[np.argmax(predictions)]
        #confidence = 100 * np.max(predictions)

    st.markdown(f"**Classe prédite :** `{predicted_class}`")
    st.markdown(f"**Confiance :** `{confidence:.2f}%`")
    #st.bar_chart(dict(zip(class_names, predictions)))
    st.markdown("### Répartition des probabilités")    
    #predictions = predictions.cpu().numpy()
    #df_probs = pd.DataFrame({
    #    "Classe": class_names,
    #    "Probabilité (%)": [round(p * 100, 2) for p in predictions]
    #}).set_index('Classe')
    #st.bar_chart(df_probs)

    df = pd.DataFrame({
        'Classe': class_names,
        'Probabilité (%)': [round(p * 100, 2) for p in predictions.cpu().numpy()]
    })

    fig = px.bar(df, x='Classe', y='Probabilité (%)', text='Probabilité (%)')

    # Inclinaison des labels à 45°
    fig.update_layout(
        xaxis_tickangle=-45,
        title="Répartition des probabilités",
        yaxis_title="Probabilité (%)",
        xaxis_title="Classe"
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Arrêt du tracker et affichage des émissions
    tracker.stop()
    st.write(f"Émissions estimées lors de l'inférence : {tracker.final_emissions*1000:.2e} g CO₂ (Estimation Code Carbone)")
