# Data

`synthetic_flow.csv` is synthetic. It is produced by `scripts/make_synthetic_data.py` with a fixed seed and contains no real trading data. Each row is one instrument on one business day:

| Column | Meaning |
|---|---|
| Date, Instrument | trading day and symbol |
| Open, High, Low, Close | simulated daily prices |
| Lots_A, Lots_B | lots traded by client segments A and B |
| PnL_A, PnL_B | the broker's P&L on each segment's flow (arbitrary units) |
| Lots_Total, PnL_Total | sums across both segments |
