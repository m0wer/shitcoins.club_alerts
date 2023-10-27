"""
Read the CSV with the past data and plot the buy and sell comission graphs.

columns: crypto_currency, fiat_currency, buy, sell, time, comission_buy, comission_sell

Dependencies:

pip install plotly pandas
"""

import os
import plotly.express as px
import pandas as pd

# Path to the csv file
CSV_PATH: str = os.environ.get("CSV_PATH", "/data/price.csv")

# Read the CSV
df = pd.read_csv(CSV_PATH)

# Plot the buy and sell comission graphs
fig = px.line(df, x="time", y=["comission_buy", "comission_sell"], title="Comissions", color="crypto_currency")
fig.show()
