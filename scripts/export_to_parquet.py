import json
from pathlib import Path
import pandas as pd

# Column descriptions from project spec
COLUMN_DESC = {
    'demand_mw': 'Original actual load (Ground Truth).',
    'trend_baseline': 'Long-term macro growth/decay extracted via STL.',
    'seasonal_diurnal': 'Repeating 24-hour daily load shape.',
    'seasonal_annual_fourier': 'Harmonic sine/cosine annual wave.',
    'baseline_point_forecast': 'Trend + Diurnal + Annual (statistical benchmark).',
    'residual_target': 'Actual - baseline_point_forecast (Primary ML Target).',
    'p10_lower_bound': 'Dynamic -1.645σ rolling residual envelope.',
    'p90_upper_bound': 'Dynamic +1.645σ rolling residual envelope.',
    'is_anomaly_flag': 'True if value outside p10/p90.',
    'is_holiday': 'Country/region-specific holiday indicator.'
}


def export_folder(csv_folder, parquet_folder):
    csv_folder = Path(csv_folder)
    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)
    schema = {}

    for csv_file in sorted(csv_folder.glob('*_processed.csv')):
        print('Exporting', csv_file.name)
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        # enforce dtypes
        df['is_anomaly_flag'] = df['is_anomaly_flag'].astype(bool)
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)

        # write parquet
        out_name = csv_file.stem + '.parquet'
        out_path = parquet_folder / out_name
        df.to_parquet(out_path, index=False)

        # collect metadata
        col_meta = []
        for c in df.columns:
            col_meta.append({
                'name': c,
                'dtype': str(df[c].dtype),
                'description': COLUMN_DESC.get(c, '')
            })

        stats = {
            'rows': int(len(df)),
            'anomaly_count': int(df['is_anomaly_flag'].astype(bool).sum()) if 'is_anomaly_flag' in df.columns else None,
            'null_counts': df.isna().sum().to_dict()
        }

        schema[str(out_path.name)] = {
            'source_csv': str(csv_file.name),
            'parquet': str(out_path.name),
            'columns': col_meta,
            'stats': stats
        }

    # write combined schema/metadata
    meta_path = parquet_folder / 'features_schema.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, default=int)

    print('Wrote parquet files to', parquet_folder)
    print('Wrote schema to', meta_path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv-folder', required=True)
    p.add_argument('--parquet-folder', required=True)
    args = p.parse_args()
    export_folder(args.csv_folder, args.parquet_folder)
