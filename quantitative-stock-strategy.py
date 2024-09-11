import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from datetime import datetime

# List of SPDR Sector ETFs
base_sector_etfs = [
    'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI',
    'XLB', 'XLK', 'XLU'
]

def fetch_data(tickers, start_date, end_date):
    """Fetch historical data for given tickers."""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close'].tz_localize(None)  # Make timezone-naive

def calculate_returns(prices, period='QE'):
    """Calculate periodic returns."""
    return prices.resample(period).last().pct_change()

def create_features(returns, economic_data):
    """Create feature set for the model."""
    features = returns.copy()
    
    # Lagged returns (1, 2, 3, 4 quarters)
    for sector in returns.columns:
        for i in range(1, 5):
            features[f'{sector}_lag_{i}Q'] = returns[sector].shift(i)
    
    # Merge with economic indicators
    features = pd.merge(features, economic_data, left_index=True, right_index=True, how='outer')
    
    # Forward fill missing values
    features = features.ffill()
    
    # Backward fill any remaining missing values at the start
    features = features.bfill()
    
    return features

def train_model(features, target):
    """Train a Random Forest model."""
    X = features.drop(columns=[target])
    y = features[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def predict_returns(model, features):
    """Predict returns using the trained model."""
    return model.predict(features)

def rank_sectors(predictions):
    """Rank sectors based on predicted returns."""
    return pd.Series(predictions).sort_values(ascending=False)

def allocate_portfolio(rankings, base_allocation=0.1, top_boost=0.025):
    """Allocate portfolio based on rankings."""
    n_sectors = len(rankings)
    allocations = pd.Series(base_allocation, index=rankings.index)
    allocations.iloc[:3] += top_boost
    allocations.iloc[-3:] -= top_boost
    return allocations / allocations.sum()

def read_economic_data(start_date, end_date):
    """Read economic data from CSV file."""
    economic_data = pd.read_csv('economic_data.csv', parse_dates=['Date'], index_col='Date')
    return economic_data.loc[start_date:end_date].tz_localize(None)  # Make timezone-naive

def run_strategy(start_date, end_date):
    # Fetch historical data
    all_sectors = base_sector_etfs + ['XLRE', 'XLC', 'SPY']
    prices = fetch_data(all_sectors, start_date, end_date)
    returns = calculate_returns(prices)
    
    print("Returns shape:", returns.shape)
    print("Returns date range:", returns.index.min(), "to", returns.index.max())
    
    # Separate S&P 500 returns
    sp500_returns = returns['SPY'].to_frame()
    sector_returns = returns.drop(columns=['SPY'])
    
    # Read economic data from CSV
    economic_data = read_economic_data(start_date, end_date)
    
    print("Economic data shape:", economic_data.shape)
    print("Economic data date range:", economic_data.index.min(), "to", economic_data.index.max())
    
    # Prepare features
    features = create_features(sector_returns, economic_data)
    
    print("Features shape:", features.shape)
    print("Features date range:", features.index.min(), "to", features.index.max())
    
    # Initialize portfolio allocations DataFrame
    portfolio_allocations = pd.DataFrame(index=features.index, columns=all_sectors[:-1])  # Exclude SPY
    
    # Train models and make predictions for each quarter
    for date in features.index:
        current_features = features.loc[:date]
        
        if date < pd.Timestamp('2016-01-01'):
            current_sectors = base_sector_etfs
        elif date < pd.Timestamp('2019-01-01'):
            current_sectors = base_sector_etfs + ['XLRE']
        else:
            current_sectors = base_sector_etfs + ['XLRE', 'XLC']
        
        models = {sector: train_model(current_features, sector) for sector in current_sectors if sector in current_features.columns}
        
        predictions = {}
        for sector, model in models.items():
            X = current_features.drop(columns=[sector])  # Drop only the target sector column
            predictions[sector] = model.predict(X.iloc[-1:])[-1]
        
        rankings = rank_sectors(predictions)
        allocations = allocate_portfolio(rankings)
        
        portfolio_allocations.loc[date, current_sectors] = allocations
    
    # Fill NaN values with 0 for XLRE before 2016 and XLC before 2019
    portfolio_allocations['XLRE'].fillna(0, inplace=True)
    portfolio_allocations['XLC'].fillna(0, inplace=True)
    
    print("Portfolio allocations shape:", portfolio_allocations.shape)
    print("Portfolio allocations date range:", portfolio_allocations.index.min(), "to", portfolio_allocations.index.max())
    
    return portfolio_allocations, sector_returns, sp500_returns

# Example usage
start_date = '2010-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

portfolio_allocations, sector_returns, sp500_returns = run_strategy(start_date, end_date)

# Save data to CSV
portfolio_allocations.to_csv('portfolio_allocations.csv')
sector_returns.to_csv('sector_returns.csv')
sp500_returns.to_csv('sp500_returns.csv')

print("Data saved to CSV files")
print(f"Shape of portfolio_allocations: {portfolio_allocations.shape}")
print("Portfolio allocations date range:", portfolio_allocations.index.min(), "to", portfolio_allocations.index.max())
print("\nFirst few rows of portfolio_allocations:")
print(portfolio_allocations.head())
print("\nLast few rows of portfolio_allocations:")
print(portfolio_allocations.tail())
