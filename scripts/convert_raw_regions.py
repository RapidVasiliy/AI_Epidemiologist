"""Конвертировать сырые xlsx (RSCF_Uncertainty/generateIncidence/input) в data/regions/<город>/.

python scripts/convert_raw_regions.py <каталог с Заб/ПЦР xlsx> [--out data/regions] [--city moscow ...]
"""
import argparse

from ai4epi.data.raw_regions import RAW_CITIES, convert_raw_city


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir")
    parser.add_argument("--out", default="data/regions")
    parser.add_argument("--city", action="append", choices=sorted(RAW_CITIES))
    parser.add_argument("--start-year", type=int, default=2010)
    args = parser.parse_args()
    cities = args.city or [c for c in RAW_CITIES if c != "spb_raw"]
    for city in cities:
        print(convert_raw_city(city, raw_dir=args.raw_dir, out_dir=args.out, start_year=args.start_year))


if __name__ == "__main__":
    main()
