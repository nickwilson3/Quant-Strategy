# Quantitative Trading Strategy
Nick Wilson - *Claremont McKenna College, Emory University (Master's of Analytical Finance)*

Jacky Tan - *Indiana University, Emory University (Master's of Analytical Finance)*

## Portfolio Summary
*Since 2010*

**Average Annual Return (Portfolio): 15.92%**

Average Annual Return (S&P 500): 13.71%

**Volatility (Portfolio): 11.97%**

Volatility (S&P 500): 14.14%

Alpha: 0.0465

Beta: 0.8224

Sharpe Ratio: 1.16

**Final value of $1000 investment in Portfolio: $8492.07**

Final value of $1000 investment in S&P 500: $6114.48

![Portfolio Performance vs. S&P 500 Performance](https://github.com/nickwilson3/Quant-Strategy/blob/main/annual_returns_comparison.png)
![1000$ Invested into Portfolio vs S&P 500](https://github.com/nickwilson3/Quant-Strategy/blob/main/investment_growth_comparison.png)
## Investment Thesis
This portfolio, powered by quantitative analysis, is essentially a rebalanced S&P 500 that is optimized by projected individual sector performance. The portfolio is a mix of SPDR S&P 500 Sector ETFS. The thesis is that by evaluating key economic factors and lagged returns of sector performance, you can reallocate the sectors to optimize the S&P 500 to produce higher returns, with less risk.

### Data
1. 11 SPDR Sector Etfs + S&P 500 - *price data imported through yfinance*
   - XLC : Communication Services
   - XLY: Consumer Discretionary
   - XLP: Consumer Staples
   - XLE: Energy
   - XLF: Financials
   - XLV: Health Care
   - XLI: Industials
   - XLB: Materials
   - XLRE: Real Estate
   - XLK: Technology
   - XLU: Utilities
   - SPY: S&P 500
2. Key Economic Factors - *data imported through FRED API*
   - GDP Growth
   - Unemployment Rate
   - Interest Rate
   - Housing Cost Index
   - Inflation Rate
   - Industrial Production
   - Consumer Sentiment
   - S&P 500
   - Yield Curve
   - Corporate Profits
  
  ### Quantitative Methods

This trading strategy employs several quantitative techniques to make sector allocation decisions:

1. **Random Forest Regression**: The core of the strategy uses Random Forest models to predict sector returns. Random Forests are an ensemble learning method that constructs multiple decision trees and outputs the average prediction of the individual trees.

2. **Feature Engineering**: The strategy creates lagged returns (1 to 4 quarters) for each sector ETF as predictive features. It also incorporates economic indicators to capture broader market conditions.

3. **Periodic Rebalancing**: The portfolio is rebalanced quarterly based on the predictions from the Random Forest models.

4. **Dynamic Sector Inclusion**: The strategy adapts to the introduction of new sector ETFs over time (XLRE in 2016 and XLC in 2019), adjusting the allocation methodology as new sectors become available.

5. **Ranking-based Allocation**: Sectors are ranked based on their predicted returns, and allocations are adjusted to overweight top-ranked sectors and underweight bottom-ranked sectors.

6. **Performance Evaluation**: The strategy's performance is evaluated against the S&P 500 benchmark using metrics such as annual returns, cumulative returns, Sharpe ratio, alpha, and beta.

7. **Time Series Analysis**: The strategy uses historical time series data of sector ETF prices and economic indicators to train the models and make predictions.

This quantitative approach aims to capture sector rotation dynamics and economic trends to potentially outperform the broad market index.
  
### Key Takeaways
- Less downside risk in 2018 and 2022
- Zero instances where S&P 500 was up and portfolio was down
- Overall, less volatily and higher returns

### Next Steps
- Implement a condition to capture companies that experience exponential growth - such as Nvidia, ELi Lily, Tesla, etc.
- Optimize rebalancing fees
