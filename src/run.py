import argparse
from .process import load_data, compute_stat_components, save_outputs


def main():
    p = argparse.ArgumentParser(description='Compute statistical baseline and residuals for energy demand')
    p.add_argument('--input', '-i', required=True, help='Input CSV with timestamp and demand_mw columns')
    p.add_argument('--output', '-o', required=True, help='Output CSV path')
    p.add_argument('--country', default='US', help='Country code for holiday calendar (default US)')
    p.add_argument('--n-harmonics', type=int, default=3, help='Number of annual harmonics')
    p.add_argument('--rolling-window', type=int, default=168, help='Rolling window hours for sigma (default 168)')
    p.add_argument('--ts-col', default='timestamp', help='Timestamp column name in input (default "timestamp")')
    p.add_argument('--value-col', default='demand_mw', help='Value column name in input (default "demand_mw")')
    args = p.parse_args()

    df = load_data(args.input, ts_col=args.ts_col, value_col=args.value_col)
    out = compute_stat_components(df, n_harmonics=args.n_harmonics, rolling_window_hours=args.rolling_window, country=args.country)
    save_outputs(out, args.output)
    print('Wrote outputs to', args.output)


if __name__ == '__main__':
    main()
