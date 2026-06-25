# Public Benchmark Strategy Suite

Initial public suite:

| ID | Strategy | Purpose |
| --- | --- | --- |
| BM01 | SMA crossover without costs | Indicator alignment and daily signal timing |
| BM02 | RSI mean reversion | Indicator definition, warmup, and threshold timing |
| BM03 | Monthly rebalance portfolio | Calendar semantics, weight drift, and rebalance timing |
| BM04 | Top-N momentum rotation | Ranking windows, volatility sizing, and partial allocation |
| BM05 | Dual momentum with cash switching | Absolute/relative momentum and cash-switch semantics |

Candidate public additions:

| ID | Strategy | Purpose |
| --- | --- | --- |
| BM06 | SMA crossover with costs | Turnover and fee accounting |
| BM07 | Long-short market neutral | Gross/net exposure and financing assumptions |
| BM08 | Stop-loss or bracket order case | Order lifecycle and same-bar ambiguity |
| BM09 | Dividend/corporate-action sensitivity | Adjusted data and cash event handling |
| BM10 | Out-of-sample split consistency | Reproducible train/test chronology |
