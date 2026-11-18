"""Свести общероссийскую выгрузку ПЦР по возрастам (export_LABD_PCR_weekly_*.xlsx) в data/national.

python scripts/convert_age_pcr_export.py export_LABD_PCR_weekly_201840_201939.xlsx [...] [--out data/national]

Имя CSV — по первой и последней неделе выгрузки (age_pcr_<год>w<нед>_<год>w<нед>.csv); загрузчик
(``ai4epi.data.age_pcr.load_age_pcr``) читает все такие файлы каталога.
"""
import argparse
from pathlib import Path

from ai4epi.data.age_pcr import extract_age_pcr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--out", default="data/national", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for export in args.exports:
        table = extract_age_pcr(export)
        first = table.sort_values(["iso_year", "iso_week"]).iloc[0]
        last = table.sort_values(["iso_year", "iso_week"]).iloc[-1]
        name = f"age_pcr_{int(first.iso_year)}w{int(first.iso_week):02d}_{int(last.iso_year)}w{int(last.iso_week):02d}.csv"
        table.to_csv(args.out / name, index=False)
        print(f"{export} -> {args.out / name}: {table['district'].nunique()} городов, {len(table)} строк")


if __name__ == "__main__":
    main()
