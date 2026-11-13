# Fixed-income-market-making-algo

A small, self-contained market-making simulation. Prices follow an
Ornstein–Uhlenbeck (mean-reverting) process; a simple market maker quotes a
fixed spread around the current price and tracks cash, inventory and profit.

Based on *Market Making and Mean Reversion* by Tanmoy Chakraborty and
Michael Kearns (credited in the source file).

## How to run

```bash
pip install -r requirements.txt
python Market_making.py
```

No data files are needed — the price series is simulated. The script saves a
plot of price vs. profit to `market_making_results.png` and also displays it.
