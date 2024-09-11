import pandas as pd
import yfinance as yf

def fetch_etf_data(tickers, start_date, end_date):
    """Fetch ETF data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

def calculate_returns(prices):
    """Calculate quarterly returns."""
    return prices.resample('Q').last().pct_change()

# Define sector ETFs and date range
sector_etfs = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU', 'XLRE', 'XLC']
sp500_ticker = '^GSPC'
start_date = '2010-01-01'
end_date = '2023-12-31'

# Fetch data
sector_prices = fetch_etf_data(sector_etfs, start_date, end_date)
sp500_prices = fetch_etf_data(sp500_ticker, start_date, end_date)

# Calculate returns
sector_returns = calculate_returns(sector_prices)
sp500_returns = calculate_returns(sp500_prices)

# Save to CSV
sector_returns.to_csv('sector_returns.csv')
sp500_returns.to_csv('sp500_returns.csv')

print("Data has been fetched, processed, and saved to 'sector_returns.csv' and 'sp500_returns.csv'")

# Display the first few rows of each dataset
print("\nSector Returns:")
print(sector_returns.head())

print("\nS&P 500 Returns:")
print(sp500_returns.head())
