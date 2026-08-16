import os
import sys
from pathlib import Path
# Ensure project src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.process import load_data, compute_stat_components, save_outputs


def batch_process(input_folder, output_folder, country='US'):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for f in input_folder.glob('*_clean.csv'):
        try:
            print('Processing', f.name)
            df = load_data(str(f), ts_col='Datetime_EPT', value_col='Demand_MW')
            out = compute_stat_components(df, country=country)
            out_path = output_folder / f.name.replace('.csv', '_processed.csv')
            save_outputs(out, str(out_path))
        except Exception as e:
            print('Failed', f.name, e)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input-folder', required=True)
    p.add_argument('--output-folder', required=True)
    p.add_argument('--country', default='US')
    args = p.parse_args()
    batch_process(args.input_folder, args.output_folder, country=args.country)
