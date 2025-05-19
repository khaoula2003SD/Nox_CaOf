# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prédictions NOx et CaO libre", layout="wide")
st.title("Prédiction de la pollution NOx et du CaO libre")

# Téléversement des fichiers CSV
st.sidebar.header("1. Chargement des fichiers CSV")
uploaded_nox = st.sidebar.file_uploader("Fichier CSV pour NOx", type="csv")
uploaded_caof = st.sidebar.file_uploader("Fichier CSV pour CaO libre", type="csv")

if not uploaded_nox or not uploaded_caof:
    st.sidebar.warning("Merci de charger les deux fichiers CSV nécessaires.")
    st.stop()

# === NOx ===
st.subheader("🟦 Prédiction NOx")

df_nox = pd.read_csv(uploaded_nox, na_values=["null", "NA"])
df_nox['date'] = pd.to_datetime(df_nox['date'], format="%d.%m.%Y %H:%M")

X_nox = df_nox.drop(columns=['date', 'Nox_baf', 'Nox opsis'])
X_nox = X_nox.apply(pd.to_numeric, errors='coerce').fillna(X_nox.mean())

# Chargement des modèles NOx
model_baf = joblib.load("Nox1_modèle.pkl")
model_opsis = joblib.load("Nox_opsis_linearregression.pkl")

# Prédictions NOx
df_nox['Nox_baf_pred'] = model_baf.predict(X_nox)
df_nox['Nox_opsis_pred'] = model_opsis.predict(X_nox)

# Alertes
def alerte(val, seuil_att, seuil_dang):
    if val >= seuil_dang: return "DANGER"
    if val >= seuil_att: return "ATTENTION"
    return "OK"

df_nox['Alerte_baf'] = df_nox['Nox_baf_pred'].apply(alerte, args=(400, 500))
df_nox['Alerte_opsis'] = df_nox['Nox_opsis_pred'].apply(alerte, args=(350, 450))
df_nox['Alerte'] = df_nox[['Alerte_opsis', 'Alerte_baf']].max(axis=1)

# Graphique des alertes
st.markdown("**Distribution des alertes NOx**")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_nox['Alerte_baf'].value_counts().plot.bar(ax=axes[0], title="BAF")
df_nox['Alerte_opsis'].value_counts().plot.bar(ax=axes[1], title="OPSIS")
st.pyplot(fig)

# Visualisation interactive NOx
st.markdown("**Visualisation temporelle**")
target = st.selectbox("Type NOx", ["BAF", "OPSIS"])
date_range = st.date_input("Plage de dates", [df_nox.date.min().date(), df_nox.date.max().date()])
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df_f = df_nox[(df_nox['date'] >= start) & (df_nox['date'] <= end)]
col_pred = 'Nox_baf_pred' if target == "BAF" else 'Nox_opsis_pred'
col_alert = 'Alerte_baf' if target == "BAF" else 'Alerte_opsis'
seuils = (400, 500) if target == "BAF" else (350, 450)

fig, ax = plt.subplots(figsize=(12, 5))
colors = {"OK": "green", "ATTENTION": "orange", "DANGER": "red"}
for lvl, c in colors.items():
    sub = df_f[df_f[col_alert] == lvl]
    ax.scatter(sub.date, sub[col_pred], label=lvl, c=c, s=10)
for s in seuils:
    ax.axhline(s, linestyle="--")
ax.legend()
ax.set_title(f"{target} prédits")
st.pyplot(fig)

st.markdown("**Tableau des résultats NOx**")
colonnes = ['date', 'Nox opsis', 'Nox_opsis_pred', 'Alerte_opsis', 'Nox_baf', 'Nox_baf_pred', 'Alerte_baf', 'Alerte']
st.dataframe(df_nox[colonnes])
csv_nox = df_nox.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger NOx", csv_nox, "resultats_nox.csv", "text/csv")

# === CaO libre ===
st.subheader("🟩 Prédiction CaO libre")

df_caof = pd.read_csv(uploaded_caof)
df_caof['date'] = pd.to_datetime(df_caof['date'], format="%d.%m.%Y %H:%M")

# Préparation des features
X_caof = df_caof.drop(columns=['date', 'CaO f', 'LS', 'SR', 'AR', 'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3'])

# Chargement des scalers
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Normalisation
X_scaled = scaler_X.transform(X_caof)

# Chargement modèle Keras
model = load_model("mon_modele.h5")
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

df_caof['CaO f prédit'] = y_pred.flatten()

st.line_chart(df_caof.set_index('date')[['CaO f', 'CaO f prédit']])
st.dataframe(df_caof[['date', 'CaO f', 'CaO f prédit']])
csv_caof = df_caof.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger CaO libre", csv_caof, "resultats_caof.csv", "text/csv")

