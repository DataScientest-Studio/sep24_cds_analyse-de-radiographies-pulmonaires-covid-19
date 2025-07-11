import streamlit as st
from utils import interactive_image  # Si nécessaire pour la démo
import random
from PIL import Image
import joblib
import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_model.json")
    return model

model = load_model()

def extract_features(image_pil):
    image = image_pil.convert("L").resize((128, 128))
    image_np = np.array(image)
    features = hog(
        image_np,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    return features, image_np 

def get_hog_image(gray_img):
    _, hog_image = hog(
        gray_img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True
    )
    # Convertir en image Streamlit-friendly
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    ax.imshow(hog_image, cmap='gray')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

st.set_page_config(page_title="Modélisation ML", layout="wide")
st.title("Modèles de Machine Learning")

data_ml = [
    {"Modèle": "KNN", "F1-score (%)": 77.0, "Accuracy (%)": 83.0, "Temps (s)": 12, "Params": 0},
    {"Modèle": "Random Forest", "F1-score (%)": 83.0, "Accuracy (%)": 86.0, "Temps (s)": 30, "Params": 10000},
    {"Modèle": "SVM", "F1-score (%)": 82.0, "Accuracy (%)": 85.0, "Temps (s)": 160, "Params": 5000},
    {"Modèle": "XGBoost", "F1-score (%)": 86.0, "Accuracy (%)": 88.0, "Temps (s)": 35, "Params": 8000},
    {"Modèle": "MLPClassifier", "F1-score (%)": 81.0, "Accuracy (%)": 84.0, "Temps (s)": 42, "Params": 150000},
]
df_ml = pd.DataFrame(data_ml)

data = {
    "Classe": ["Normal", "Covid", "Viral pneumonia", "Lung opacity", "macro avg", "weighted avg"],
    "32": [0.81, 0.80, 0.91, 0.75, 0.82, 0.83],
    "64": [0.84, 0.83, 0.93, 0.79, 0.85, 0.86],
    "128": [0.85, 0.85, 0.95, 0.80, 0.86, 0.87],
    "raw": [0.79, 0.78, 0.90, 0.76, 0.81, 0.82],
    "standard": [0.80, 0.79, 0.91, 0.78, 0.82, 0.84],
    "minmax": [0.80, 0.81, 0.92, 0.77, 0.83, 0.84],
    "robust": [0.83, 0.84, 0.94, 0.78, 0.85, 0.86],
}
df = pd.DataFrame(data)
df = df.melt(id_vars=["Classe"], var_name="Dataset", value_name="F1-Score")

st.markdown("""
### Objectif 
Utiliser des **modèles de Machine Learning classiques** pour détecter automatiquement les cas positifs sur des radiographies pulmonaires.
            
---

### Prétraitements et optimisations
- **Hyperparamètres** : Recherche automatique des meilleurs hyperparamètres (Grid search, Bayesian search)
- **Taille des images** : tests en 32×32, 64×64, 128×128
- **Standardisation** : Mise à l’échelle des données pour éviter les biais
""")
fig = px.bar(df, x="Classe", y="F1-Score", color="Dataset", barmode="group",
             title="MLP Classifier - F1-Score par classe")

st.plotly_chart(fig, use_container_width=True)


st.markdown("- **HOG** : (Histogramme de gradient orienté) Extraction de caractéristiques visuelles (bords, textures)")
image_dir = "src/streamlit/images"

if st.button("🔀 Changer les images"):
    st.session_state["refresh_images"] = True

if "refresh_images" not in st.session_state:
    st.session_state["refresh_images"] = True    

classes = ["COVID-", "Lung_Opacity-", "Normal-", "Viral Pneumonia-"]

if st.session_state["refresh_images"]:
    selected_images = []
    for cls_prefix in classes:
        matches = [f for f in os.listdir(image_dir) if f.startswith(cls_prefix)]
        if matches:
            selected = random.choice(matches)
            selected_images.append(os.path.join(image_dir, selected))
    st.session_state["selected_images"] = selected_images
    st.session_state["refresh_images"] = False

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Images originales et leurs HOG", fontsize=16)

# Affichage
cols = st.columns(4)
for i, path in enumerate(st.session_state["selected_images"]):
    img = imread(path)
    gray = rgb2gray(img) if img.ndim == 3 else img

    # Image originale
    cols[i].image(img, caption=f"{classes[i][:-1]} - originale", use_container_width=True)

    # HOG
    hog_buf = get_hog_image(gray)
    cols[i].image(hog_buf, caption=f"{classes[i][:-1]} - HOG", use_container_width=True)

st.markdown("""
            
---
| Modèle           | Pertinence | Commentaire rapide |
|------------------|------------|---------------------|
| KNN              | ❌         | Ne gère pas bien les images |
| Random Forest    | ❌         | Surfit / peu discriminant |
| SVM              | ✅         | Performant et stable |
| XGBoost          | ⭐         | Meilleur compromis |
| MLP / Voting     | 🟡         | OK mais sans gain |
                             
""")

st.dataframe(df_ml.style.format({
    "Params": "{:,.0f}",
    "Temps (s)": "{:.0f}",
    "Accuracy (%)": "{:.2f}",
    "F1-score (%)": "{:.2f}",
}), hide_index=True)


import plotly.graph_objects as go

# Données
models = df_ml["Modèle"]
accuracy = df_ml["Accuracy (%)"]
f1_score = df_ml["F1-score (%)"]

# Création du graphe
fig_combo = go.Figure()

# Barres : Accuracy
fig_combo.add_trace(go.Bar(
    x=models,
    y=accuracy,
    name="Accuracy (%)",
    marker_color="rgb(0, 102, 204)",
    text=accuracy,
    opacity=0.7
))

# Barres : F1-score
fig_combo.add_trace(go.Bar(
    x=models,
    y=f1_score,
    name="F1-score (%)",
    marker_color="rgb(102, 178, 255)",
    text=f1_score,
    opacity=0.7
))

# Ligne : Accuracy
fig_combo.add_trace(go.Scatter(
    x=models,
    y=accuracy,
    name="Tendance Accuracy",
    mode="lines+markers",
    line=dict(color='blue', dash='dash'),
    marker=dict(symbol='circle', size=8)
))

# Ligne : F1-score
fig_combo.add_trace(go.Scatter(
    x=models,
    y=f1_score,
    name="Tendance F1-score",
    mode="lines+markers",
    line=dict(color='green', dash='dash'),
    marker=dict(symbol='square', size=8)
))

# Mise en forme
fig_combo.update_layout(
    title="Scores Accuracy & F1-score (barres + courbes)",
    barmode="group",
    xaxis_title="Modèle",
    yaxis_title="Score (%)",
    title_x=0.3,
    legend_title="Métrique",
    height=500
)

st.plotly_chart(fig_combo, use_container_width=True)

### Focus : XGBoost

st.markdown("""
### Focus sur XGBoost
            
- Algorithme de **boosting** très efficace
- Corrige les erreurs au fur et à mesure
- Bon compromis entre performance, rapidité, et simplicité

- Entraîné sur HOG, pas besoin de normalisation
- ~88 % de F1-score
- Rapide à entraîner, prédire
- Stable sur toutes les classes
""")       


st.image("src/images/xgboost_matrice.png")

st.markdown("---")
st.subheader("Tester une image avec XGBoost")

test_samples = {
    "Normal": "src/streamlit/images/xgb-normal.png",
    "COVID": "src/streamlit/images/xgb-covid.png",
    "Pneumonie": "src/streamlit/images/xgb-pneumonia.png"
}

class_names = ["Normal", "Covid", "Pneumonie", "Opacité pulmonaire"]

CLASS_PREFIXES = {
    "COVID": "COVID-",
    "Opacité pulmonaire": "Lung_Opacity-",
    "Normal": "Normal-",
    "Pneumonie": "Viral Pneumonia-"
}
CLASS_NAMES = list(CLASS_PREFIXES.keys())

if st.button("Rafraîchir les échantillons"):
    st.session_state["test_samples"] = None

if "test_samples" not in st.session_state or st.session_state["test_samples"] is None:
    selected_images = []
    for cls_prefix in classes:
        matches = [f for f in os.listdir(image_dir) if f.startswith(cls_prefix)]
        if matches:
            image = random.choice(matches)
            st.session_state["selected_samples"] = image
            selected_images.append(os.path.join(image_dir, image))
    st.session_state["selected_samples"] = selected_images
    st.session_state["test_samples"] = False

# Chargement des échantillons sélectionnés
test_samples = st.session_state["test_samples"]    

cols = st.columns(4)
for i, path in enumerate(st.session_state["selected_samples"]):
    with cols[i]:
        st.markdown(f"### {CLASS_NAMES[i]}")
        image = Image.open(path)
        st.image(image, caption="Image originale", use_container_width=True)

        features, gray_img = extract_features(image)

        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0]

        hog_buf = get_hog_image(gray_img)
        st.image(hog_buf, caption="HOG", use_container_width=True)

        st.markdown(f"**Prédiction :** `{CLASS_NAMES[prediction]}`")
        st.markdown("**Probabilités :**")
        st.bar_chart(dict(zip(CLASS_NAMES, proba)))

        st.markdown("**Contributions des features**")
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]

        top_features = pd.DataFrame({
            "Feature": [f"HOG {i}" for i in top_indices],
            "Importance": importances[top_indices]
        })

        st.dataframe(top_features, use_container_width=True)


st.markdown("""
---
## **Conclusion – Modèles ML classiques**

- Les modèles classiques (SVM, XGBoost) donnent de **bons résultats** après :
  - une **extraction HOG** 
  - une **optimisation des hyperparamètres** adaptée (GridSearch / tuning manuel)

- **XGBoost** se démarque comme le **meilleur compromis** :
  - robuste aux variations de **taille**, **normalisation**, ou **prétraitement**
  - rapide à entraîner
  - stable sur toutes les classes (≈88 % F1-score)

""")
