import streamlit as st
from utils import interactive_image

st.set_page_config(page_title="Modélisation", layout="wide")


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
st.image("src/images/TablaModelos.png", caption="exemple", width=750)

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
