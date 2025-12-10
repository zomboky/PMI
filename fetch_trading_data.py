import yfinance as yf
import pandas as pd

# --- Paramètres ---
SYMBOL = "OJ=F"  # Orange Juice Futures
START_DATE = "2015-08-01"
END_DATE   = "2023-12-28"
INTERVAL = "1d"  # Toutes les 5 jours

# --- Télécharger les données ---
data = yf.download(SYMBOL, start=START_DATE, end=END_DATE, interval=INTERVAL)

# --- Garder uniquement Date et Close ---
prices = data[['Close']].reset_index()
prices.columns = ['date', 'price_per_lb']  # renommer colonnes

# --- Exporter CSV ---
prices.to_csv("orange_juice_futures_brazil_proxy.csv", index=False)

print("✔ Done! Aperçu des données :")
print(prices.head())



