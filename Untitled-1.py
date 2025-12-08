# %%
import pandas as pd

# Lire le CSV

df = pd.read_csv('data/Cinturao_10ans_daily_1km.csv')

# Convertir la colonne 'date' en datetime
df['datetime'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Trier le DataFrame par date (optionnel)
df = df.sort_values('datetime').reset_index(drop=True)

# Sélectionner uniquement les colonnes utiles
cols = ['datetime', 'NDVI', 'EVI', 'NDWI', 'NDMI', 'B2','B3','B4','B8','B11']
df = df[cols]

# Afficher le résultat
df.head()

# %% [markdown]
# # Description des colonnes du CSV exporté depuis Google Earth Engine
# 
# Le CSV contient les valeurs des indices agricoles et des bandes spectrales pour chaque carré de 1 km² et chaque date. Voici la signification de chaque colonne :
# 
# | Colonne       | Description |
# |---------------|-------------|
# | `system:index`| Identifiant unique de l'image Sentinel-2 pour cette date et scène. Format : `YYYYMMDDTHHMMSS_...`. |
# | `B2`          | Bande 2 de Sentinel-2 (Blue, 490 nm), valeurs de réflectance en surface. |
# | `B3`          | Bande 3 de Sentinel-2 (Green, 560 nm), valeurs de réflectance en surface. |
# | `B4`          | Bande 4 de Sentinel-2 (Red, 665 nm), valeurs de réflectance en surface. |
# | `B8`          | Bande 8 de Sentinel-2 (NIR, 842 nm), valeurs de réflectance en surface. |
# | `B11`         | Bande 11 de Sentinel-2 (SWIR1, 1610 nm), valeurs de réflectance en surface. |
# | `NDVI`        | Normalized Difference Vegetation Index : `(NIR - RED) / (NIR + RED)` <br> Indique la vigueur de la végétation (0 à 1 = végétation faible à forte). |
# | `EVI`         | Enhanced Vegetation Index : formule améliorée pour les zones à forte densité végétale. <br> Valeurs typiques : 0 à 1 (valeurs plus élevées = végétation plus dense). |
# | `NDWI`        | Normalized Difference Water Index : `(GREEN - NIR) / (GREEN + NIR)` <br> Indique la teneur en eau des plantes ou du sol. |
# | `NDMI`        | Normalized Difference Moisture Index : `(NIR - SWIR1) / (NIR + SWIR1)` <br> Indique l'humidité de la végétation. |
# | `date`        | Date de l'image, au format `YYYY-MM-DD`. |
# | `grid_id`     | Identifiant du carré de 1 km² correspondant dans la zone d'étude. |
# | `.geo`        | Coordonnées géométriques du polygone représentant le carré dans la zone d'étude (GeoJSON). |
# 
# > ⚠️ Les valeurs des bandes (B2, B3, B4, B8, B11) sont exprimées en **réflectance de surface**, généralement multipliées par 10000 selon GEE.  
# > Les indices (NDVI, EVI, NDWI, NDMI) sont des valeurs **normalisées** généralement comprises entre -1 et 1, ou 0 et 1 selon l’indice.
# 

# %%
df.columns.tolist()

# %%
df.head()

# %%
df.to_csv('data/Cinturao_10ans_daily_1km_sorted.csv', index=False)

# %%
df.isna().sum()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assurer que la colonne datetime est bien au format datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Calculer la moyenne quotidienne de tous les indices sur tous les carrés
daily_mean = df.groupby('datetime')[['NDVI','EVI','NDWI','NDMI']].mean().reset_index()

# Paramètres de style
sns.set(style='whitegrid', palette='muted', context='notebook', font_scale=1.2)

# Tracer l'évolution
plt.figure(figsize=(15,6))
plt.plot(daily_mean['datetime'], daily_mean['NDVI'], label='NDVI', color='green')
plt.plot(daily_mean['datetime'], daily_mean['EVI'], label='EVI', color='blue')
plt.plot(daily_mean['datetime'], daily_mean['NDWI'], label='NDWI', color='cyan')
plt.plot(daily_mean['datetime'], daily_mean['NDMI'], label='NDMI', color='brown')

plt.xlabel('Date')
plt.ylabel('Indice moyen')
plt.title('Évolution quotidienne moyenne des indices sur tous les carrés')
plt.legend()
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assurer que la colonne datetime est bien au format datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Supprimer les lignes contenant des NaN dans les colonnes d'indices
df_clean = df.dropna(subset=['NDVI','EVI','NDWI','NDMI'])

# Calculer la moyenne quotidienne de tous les indices sur tous les carrés
daily_mean = df_clean.groupby('datetime')[['NDVI','EVI','NDWI','NDMI']].mean().reset_index()

# Paramètres de style
sns.set(style='whitegrid', palette='muted', context='notebook', font_scale=1.2)

# Tracer l'évolution
plt.figure(figsize=(15,6))
plt.plot(daily_mean['datetime'], daily_mean['NDVI'], label='NDVI', color='green')
plt.plot(daily_mean['datetime'], daily_mean['EVI'], label='EVI', color='blue')
plt.plot(daily_mean['datetime'], daily_mean['NDWI'], label='NDWI', color='cyan')
plt.plot(daily_mean['datetime'], daily_mean['NDMI'], label='NDMI', color='brown')

plt.xlabel('Date')
plt.ylabel('Indice moyen')
plt.title('Évolution quotidienne moyenne des indices sur tous les carrés (sans NaN)')
plt.legend()
plt.tight_layout()
plt.show()


# %%
df.isna().sum()

# %%
from scipy import stats

# Calculer le z-score pour chaque indice
z_scores = df_clean[['NDVI','EVI','NDWI','NDMI']].apply(stats.zscore)

# Définir un seuil pour détecter les outliers (ex: |z| > 3)
threshold = 1
mask = (z_scores.abs() < threshold).all(axis=1)

# Garder uniquement les valeurs non-outliers
df_filtered = df_clean[mask]

# Calculer la moyenne quotidienne
daily_mean = df_filtered.groupby('datetime')[['NDVI','EVI','NDWI','NDMI']].mean().reset_index()

# Tracer les courbes
sns.set(style='whitegrid', palette='muted', context='notebook', font_scale=1.2)

plt.figure(figsize=(15,6))
plt.plot(daily_mean['datetime'], daily_mean['NDVI'], label='NDVI', color='green')
plt.plot(daily_mean['datetime'], daily_mean['EVI'], label='EVI', color='blue')
plt.plot(daily_mean['datetime'], daily_mean['NDWI'], label='NDWI', color='cyan')
plt.plot(daily_mean['datetime'], daily_mean['NDMI'], label='NDMI', color='brown')

plt.xlabel('Date')
plt.ylabel('Indice moyen')
plt.title('Évolution quotidienne moyenne des indices (outliers supprimés)')
plt.legend()
plt.tight_layout()
plt.show()

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df_clean = df.dropna(subset=['NDVI','EVI','NDWI','NDMI']).sort_values('datetime').reset_index(drop=True)


# Ici, pour la prévision future, on utilise EVI, NDWI, NDMI comme features pour prédire NDVI
X_all = df[['EVI','NDWI','NDMI']]
y_all = df['NDVI']

# -------------------------
# 3️⃣ Séparer train/validation/test (chronologique)
# -------------------------
n = len(df)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

X_train, y_train = X_all.iloc[:train_end], y_all.iloc[:train_end]
X_val, y_val = X_all.iloc[train_end:val_end], y_all.iloc[train_end:val_end]
X_test, y_test = X_all.iloc[val_end:], y_all.iloc[val_end:]

# -------------------------
# 4️⃣ Entraîner le modèle sur toutes les données passées
# -------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# -------------------------
# 5️⃣ Prévoir toutes les valeurs futures (validation + test)
# -------------------------
X_future = pd.concat([X_val, X_test])
y_pred_future = rf.predict(X_future)

# -------------------------
# 6️⃣ Évaluation
# -------------------------
y_true_future = pd.concat([y_val, y_test]).values

rmse = mean_squared_error(y_true_future, y_pred_future, squared=False)
mae = mean_absolute_error(y_true_future, y_pred_future)
r2 = r2_score(y_true_future, y_pred_future)

print(f"Forecast Future - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# -------------------------
# 7️⃣ Visualisation
# -------------------------
sns.set(style='whitegrid', context='notebook', font_scale=1.2)

plt.figure(figsize=(15,6))
plt.plot(df['datetime'].iloc[train_end:], y_true_future, label='Observed Future', color='blue')
plt.plot(df['datetime'].iloc[train_end:], y_pred_future, label='Predicted Future', color='orange', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.title('Random Forest - Observed vs Predicted NDVI (Futures)')
plt.legend()
plt.tight_layout()
plt.show()




# %%
df.isna().sum()


