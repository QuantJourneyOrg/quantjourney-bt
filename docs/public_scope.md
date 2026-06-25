# Public Light Scope

The public repository is a controlled light release surface for QuantJourney backtester examples and benchmark code.

Included:

- Target-weight and order-mode strategy examples.
- Import checks that run without credentials.
- Real backtests that run with QuantJourney API credentials.
- Benchmark strategy source files for comparison pages.
- Small tests for public API behavior.
- Public report artifacts built from the native QuantJourney metric and plotting modules: text summary, JSON/CSV metrics, equity CSV, dashboard HTML, equity PNG, selected PNG chart pack and run metadata.

Excluded:

- Full PDF factsheets and institutional report packets.
- Pickle archives by default; `portfolio_data.pkl`, `instruments_data.pkl` and `blotter.pkl` are opt-in local debug artifacts.
- Pro-only diagnostics: full plot orchestration, crisis analysis, trace plots, blotter plots and narrative generation.
- Walk-forward validation and optimization.
- Private cloud credential material.
- Deployment, registry and production infrastructure.
- Internal-only research orchestration code.
- Internal report publication workflows.

Release rule: public code must install from a clean checkout and pass import/tests without private services. Real backtests may require a QuantJourney account or API key.
