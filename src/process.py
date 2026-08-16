import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from dateutil import parser
import holidays


def load_data(path, ts_col='timestamp', value_col='demand_mw'):
    df = pd.read_csv(path)
    # robust parsing: try fast vectorized parse, fall back to dateutil for bad rows
    try:
        parsed = pd.to_datetime(df[ts_col], utc=True, errors='coerce', infer_datetime_format=True)
    except Exception:
        parsed = pd.to_datetime(df[ts_col], utc=True, errors='coerce')

    # fallback for any NaT using dateutil
    if parsed.isna().any():
        mask = parsed.isna()
        parsed_loc = df.loc[mask, ts_col].apply(lambda x: parser.parse(x) if pd.notna(x) else pd.NaT)
        parsed.loc[mask] = pd.to_datetime(parsed_loc, utc=True)

    # convert to US/Eastern and drop tz info for consistent hourly indexing
    try:
        parsed = parsed.dt.tz_convert('US/Eastern').dt.tz_localize(None)
    except Exception:
        # if parsing produced tz-naive timestamps, leave as-is
        parsed = parsed.dt.tz_localize(None)

    df[ts_col] = parsed
    # select only the value column for numeric processing
    df_val = df[[ts_col, value_col]].copy()
    df_val = df_val.set_index(ts_col)
    # if duplicates exist, aggregate by mean (common for meter aggregation)
    if df_val.index.duplicated().any():
        df_val = df_val.groupby(df_val.index).mean()
    df_val = df_val.sort_index()
    return df_val.rename(columns={value_col: 'demand_mw'})


def ensure_hourly_index(df):
    start = df.index.min()
    end = df.index.max()
    idx = pd.date_range(start=start, end=end, freq='h')
    df = df.reindex(idx)
    return df


def fit_annual_fourier(y, index, n_harmonics=3):
    # index is DatetimeIndex in hours
    hours_in_year = 24.0 * 365.25
    t = (index.view('int64') // 10**9) / 3600.0  # hours since epoch
    omega = 2 * np.pi / hours_in_year
    X = []
    for k in range(1, n_harmonics + 1):
        X.append(np.sin(k * omega * t))
        X.append(np.cos(k * omega * t))
    X = np.column_stack(X)
    # solve least squares for coefficients
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    annual = X.dot(coeffs)
    return annual


def compute_stat_components(df, n_harmonics=3, rolling_window_hours=168, country='US'):
    df = ensure_hourly_index(df)
    # interpolate small gaps
    df['demand_mw'] = df['demand_mw'].interpolate(method='time')

    # STL for daily seasonality + trend
    stl = STL(df['demand_mw'], period=24, robust=True)
    res = stl.fit()
    df['trend_baseline'] = res.trend
    df['seasonal_diurnal'] = res.seasonal

    # annual Fourier
    y_detrend_diurnal = df['demand_mw'] - df['trend_baseline'] - df['seasonal_diurnal']
    annual = fit_annual_fourier(y_detrend_diurnal.fillna(0).values, df.index, n_harmonics=n_harmonics)
    df['seasonal_annual_fourier'] = annual

    # baseline point forecast
    df['baseline_point_forecast'] = df['trend_baseline'] + df['seasonal_diurnal'] + df['seasonal_annual_fourier']

    # residual target
    df['residual_target'] = df['demand_mw'] - df['baseline_point_forecast']

    # rolling sigma of residuals
    df['rolling_sigma'] = df['residual_target'].rolling(window=rolling_window_hours, min_periods=24).std()

    # p10/p90 bounds around baseline
    df['p10_lower_bound'] = df['baseline_point_forecast'] - 1.645 * df['rolling_sigma']
    df['p90_upper_bound'] = df['baseline_point_forecast'] + 1.645 * df['rolling_sigma']

    # anomaly flag
    df['is_anomaly_flag'] = ((df['demand_mw'] > df['p90_upper_bound']) | (df['demand_mw'] < df['p10_lower_bound'])).fillna(False)

    # holiday indicator
    try:
        cal = holidays.CountryHoliday(country)
        df['is_holiday'] = df.index.normalize().map(lambda d: 1 if d in cal else 0)
    except Exception:
        df['is_holiday'] = 0

    # keep only required output columns
    out_cols = ['demand_mw', 'trend_baseline', 'seasonal_diurnal', 'seasonal_annual_fourier',
                'baseline_point_forecast', 'residual_target', 'p10_lower_bound', 'p90_upper_bound',
                'is_anomaly_flag', 'is_holiday']

    return df[out_cols]


def save_outputs(df, out_path):
    df.to_csv(out_path, index_label='timestamp')
