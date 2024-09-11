import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
import yfinance as yf

# You'll need to sign up for a FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = 'a36a50eef47644cdc4f920c66e026250'

fred = Fred(api_key=FRED_API_KEY)

def get_fred_data(series_id, start_date, end_date):
    """Fetch data from FRED."""
    data = fred.get_series(series_id, start_date, end_date)
    return data.resample('Q').last()  # Resample to quarterly data

def get_gdp_growth(start_date, end_date):
    """Calculate GDP growth rate."""
    gdp = get_fred_data('GDP', start_date, end_date)
    gdp_growth = gdp.pct_change() * 100
    return gdp_growth

def get_unemployment_rate(start_date, end_date):
    """Fetch unemployment rate."""
    return get_fred_data('UNRATE', start_date, end_date)

def get_interest_rate(start_date, end_date):
    """Fetch effective federal funds rate."""
    return get_fred_data('FEDFUNDS', start_date, end_date)

def get_housing_cost_index(start_date, end_date):
    """Fetch housing cost index (using Case-Shiller Home Price Index)."""
    return get_fred_data('CSUSHPINSA', start_date, end_date)

def get_inflation_rate(start_date, end_date):
    """Calculate inflation rate using CPI."""
    cpi = get_fred_data('CPIAUCSL', start_date, end_date)
    inflation = cpi.pct_change() * 100
    return inflation

def get_industrial_production(start_date, end_date):
    """Fetch industrial production index."""
    return get_fred_data('INDPRO', start_date, end_date)

def get_consumer_sentiment(start_date, end_date):
    """Fetch consumer sentiment index."""
    return get_fred_data('UMCSENT', start_date, end_date)

def get_sp500_returns(start_date, end_date):
    """Fetch S&P 500 returns."""
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    return sp500.resample('Q').last().pct_change() * 100

def get_yield_curve(start_date, end_date):
    """Fetch yield curve (10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity)."""
    return get_fred_data('T10Y2Y', start_date, end_date)

def get_corporate_profits(start_date, end_date):
    """Fetch corporate profits after tax."""
    profits = get_fred_data('CPATAX', start_date, end_date)
    return profits.pct_change() * 100  # Convert to percentage change

def create_economic_dataset(start_date, end_date):
    """Create a comprehensive economic dataset."""
    data = pd.DataFrame({
        'GDP_Growth': get_gdp_growth(start_date, end_date),
        'Unemployment_Rate': get_unemployment_rate(start_date, end_date),
        'Interest_Rate': get_interest_rate(start_date, end_date),
        'Housing_Cost_Index': get_housing_cost_index(start_date, end_date),
        'Inflation_Rate': get_inflation_rate(start_date, end_date),
        'Industrial_Production': get_industrial_production(start_date, end_date),
        'Consumer_Sentiment': get_consumer_sentiment(start_date, end_date),
        'SP500_Returns': get_sp500_returns(start_date, end_date),
        'Yield_Curve': get_yield_curve(start_date, end_date),
        'Corporate_Profits_Growth': get_corporate_profits(start_date, end_date)
    })
    
    # Ensure the index is named 'Date'
    data.index.name = 'Date'
    
    return data.dropna()

# Set date range
start_date = '2010-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Create and save the dataset
economic_data = create_economic_dataset(start_date, end_date)
economic_data.to_csv('economic_data.csv', index=True)  # Add index=True to include the Date column
print(economic_data.head())
print(f"\nDataset shape: {economic_data.shape}")
