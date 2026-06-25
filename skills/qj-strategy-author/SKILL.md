# QuantJourney Strategy Author

Use this skill when writing public QuantJourney backtesting examples.

Rules:

- Keep examples focused on strategy logic.
- Use deterministic sample data unless the user explicitly requests external data.
- Avoid credentials, private endpoints, deployment scripts, and internal services.
- Make signal timing explicit.
- State cost assumptions in basis points when costs are included.
- Prefer target weights for public examples; use order semantics only when the strategy specifically tests execution behavior.

Expected pattern:

```python
prices = sample_prices([...])
signals = ...
weights = ...
result = backtest_weights(prices, weights, fee_bps=...)
```
