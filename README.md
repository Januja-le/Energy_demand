# Energy-demand statistical baseline pipeline

This small pipeline computes the following columns from an hourly CSV input with `timestamp` and `demand_mw`:

- `trend_baseline`: long-term trend via STL
- `seasonal_diurnal`: 24-hour seasonal component via STL
- `seasonal_annual_fourier`: annual harmonic reconstruction (Fourier)
- `baseline_point_forecast`: sum of trend + diurnal + annual
- `residual_target`: `demand_mw - baseline_point_forecast`
- `p10_lower_bound` / `p90_upper_bound`: dynamic ±1.645σ rolling envelope
- `is_anomaly_flag`: True if value outside envelope
- `is_holiday`: country-specific holiday indicator

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python -m src.run -i input.csv -o outputs.csv --country US
```

The output CSV contains the original `demand_mw` and all derived columns.
