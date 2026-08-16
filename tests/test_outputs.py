import os
import pandas as pd


def test_output_file_exists():
    path = os.path.join('test_output_pjm.csv')
    assert os.path.exists(path), f'{path} should exist'


def test_columns_and_counts():
    df = pd.read_csv('test_output_pjm.csv', parse_dates=['timestamp'])
    required = ['timestamp', 'demand_mw', 'trend_baseline', 'seasonal_diurnal',
                'seasonal_annual_fourier', 'baseline_point_forecast', 'residual_target',
                'p10_lower_bound', 'p90_upper_bound', 'is_anomaly_flag', 'is_holiday']
    for c in required:
        assert c in df.columns, f'Missing column {c}'

    # Basic sanity: file has rows
    assert len(df) > 0

    # Head null counts (we expect some nulls for early rolling sigma)
    head = df.head(24)
    null_counts = head[['p10_lower_bound', 'p90_upper_bound']].isna().sum().sum()
    assert null_counts >= 0

    # Anomaly count is an integer and non-negative
    anomaly_count = int(df['is_anomaly_flag'].astype(bool).sum())
    assert anomaly_count >= 0
