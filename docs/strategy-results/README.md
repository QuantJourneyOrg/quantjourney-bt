# Published strategy result previews

This directory contains one generated cumulative-return PNG for each of the 45
strategy examples that produced a standard performance report in the validated
July 11, 2026 batch. The image filename matches the public strategy route, for
example:

```text
example-weights-01-sma-daily.png
https://backtester.quantjourney.cloud/strategies/example-weights-01-sma-daily
```

The complete dashboards publish the setup, metrics, run metadata, logs and the
larger plot pack. These compact previews are kept in GitHub so code review does
not depend on an external image for its first visual check.

WF01-WF05 are workflow examples. They publish diagnostics and logs rather than
borrowing a cumulative-return image from an optimization child run.

The examples and images are educational research artifacts, not investment
recommendations or claims of future performance.
