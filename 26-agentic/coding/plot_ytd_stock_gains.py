# filename: plot_ytd_stock_gains.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbols
tickers = ['NVDA', 'TSLA']

# Define the start and end dates
start_date = '2024-01-01'
end_date = '2024-05-31'

# Fetch the stock data for the specified date range
stock_data = yf.download(tickers, start=start_date, end=end_date)

# Calculate the YTD gains
# We take the 'Adj Close' prices for calculations
adj_close = stock_data['Adj Close']

# Calculate YTD percentage gain
ytd_gain = (adj_close.loc[end_date] / adj_close.loc[start_date] - 1) * 100

# Plot the YTD gains
plt.figure(figsize=(10, 6))
ytd_gain.plot(kind='bar')
plt.title(f'YTD Stock Gains for NVDA and TSLA as of {end_date}')
plt.xlabel('Stock Ticker')
plt.ylabel('YTD Gain (%)')
plt.xticks(rotation=0)

# Save the plot to a file
plt.savefig('ytd_stock_gains.png')

print("Plot saved as 'ytd_stock_gains.png'")