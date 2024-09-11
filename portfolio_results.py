import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def make_timezone_naive(df):
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def print_df_info(df, name):
    print(f"\n{name} info:")
    print(f"Shape: {df.shape}")
    print(f"Index: {df.index}")
    print(f"Columns: {df.columns}")
    print(f"First few rows:\n{df.head()}")
    print(f"Data types:\n{df.dtypes}")

# Load data
print("Loading data...")
portfolio_allocations = pd.read_csv('portfolio_allocations.csv', index_col=0, parse_dates=True)
sector_returns = pd.read_csv('sector_returns.csv', index_col=0, parse_dates=True)
sp500_returns = pd.read_csv('sp500_returns.csv', index_col=0, parse_dates=True)

# Make all DataFrames timezone-naive
portfolio_allocations = make_timezone_naive(portfolio_allocations)
sector_returns = make_timezone_naive(sector_returns)
sp500_returns = make_timezone_naive(sp500_returns)

print_df_info(portfolio_allocations, "Portfolio Allocations")
print_df_info(sector_returns, "Sector Returns")
print_df_info(sp500_returns, "S&P 500 Returns")

# Align data and calculate portfolio returns
common_dates = portfolio_allocations.index.intersection(sector_returns.index).intersection(sp500_returns.index)
portfolio_allocations = portfolio_allocations.loc[common_dates]
sector_returns = sector_returns.loc[common_dates]
sp500_returns = sp500_returns.loc[common_dates]

# Calculate portfolio returns
portfolio_returns = (portfolio_allocations * sector_returns).sum(axis=1)

# Convert to annual returns
annual_portfolio_returns = portfolio_returns.groupby(portfolio_returns.index.year).apply(lambda x: (1 + x).prod() - 1)
annual_sp500_returns = sp500_returns.groupby(sp500_returns.index.year).apply(lambda x: (1 + x).prod() - 1).iloc[:, 0]

# Ensure both series have the same index
common_years = annual_portfolio_returns.index.intersection(annual_sp500_returns.index)
annual_portfolio_returns = annual_portfolio_returns.loc[common_years]
annual_sp500_returns = annual_sp500_returns.loc[common_years]

# Calculate average returns
avg_portfolio_return = annual_portfolio_returns.mean()
avg_sp500_return = annual_sp500_returns.mean()

# Calculate alpha and beta
X = annual_sp500_returns.values.reshape(-1, 1)
y = annual_portfolio_returns.values
slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
beta = slope
alpha = intercept

# Create a grid comparing returns
comparison = pd.DataFrame({
    'Portfolio': annual_portfolio_returns,
    'S&P 500': annual_sp500_returns
})

# Plot the annual returns comparison (without dots)
plt.figure(figsize=(12, 8))
plt.plot(comparison.index, comparison['Portfolio'], label='Portfolio')
plt.plot(comparison.index, comparison['S&P 500'], label='S&P 500')
plt.title('Annual Returns: Portfolio vs S&P 500')
plt.xlabel('Year')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.savefig('annual_returns_comparison.png')
plt.close()

# Calculate cumulative returns
cumulative_portfolio_returns = (1 + portfolio_returns).cumprod()
cumulative_sp500_returns = (1 + sp500_returns.iloc[:, 0]).cumprod()

# Calculate growth of $1000 investment
portfolio_growth = 1000 * cumulative_portfolio_returns
sp500_growth = 1000 * cumulative_sp500_returns

# Plot the growth of $1000 investment
plt.figure(figsize=(12, 8))
plt.plot(portfolio_growth.index, portfolio_growth, label='Portfolio')
plt.plot(sp500_growth.index, sp500_growth, label='S&P 500')
plt.title('Growth of $1000 Investment: Portfolio vs S&P 500')
plt.xlabel('Date')
plt.ylabel('Value ($)')
plt.legend()
plt.grid(True)
plt.savefig('investment_growth_comparison.png')
plt.close()

# Calculate additional metrics
portfolio_volatility = annual_portfolio_returns.std()
sp500_volatility = annual_sp500_returns.std()
sharpe_ratio = (avg_portfolio_return - 0.02) / portfolio_volatility  # Assuming 2% risk-free rate

# Print summary statistics
print("\nSummary Statistics:")
print(f"Average Annual Return (Portfolio): {avg_portfolio_return:.2%}")
print(f"Average Annual Return (S&P 500): {avg_sp500_return:.2%}")
print(f"Volatility (Portfolio): {portfolio_volatility:.2%}")
print(f"Volatility (S&P 500): {sp500_volatility:.2%}")
print(f"Alpha: {alpha:.4f}")
print(f"Beta: {beta:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Print final values of $1000 investment
print(f"\nFinal value of $1000 investment in Portfolio: ${portfolio_growth.iloc[-1]:.2f}")
print(f"Final value of $1000 investment in S&P 500: ${sp500_growth.iloc[-1]:.2f}")

# Save results to CSV
comparison.to_csv('annual_returns_comparison.csv')
print("\nResults have been saved to 'annual_returns_comparison.csv'")
