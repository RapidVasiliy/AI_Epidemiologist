"""Конвертировать сырые xlsx («Заб <город>.xlsx», «ПЦР <город>.xlsx») в data/regions/<код>/.

Известные города RSCF_Uncertainty:
    python scripts/convert_raw_regions.py --raw-dir <каталог> [--city moscow ...]
Любой другой город в том же формате:
    python scripts/convert_raw_regions.py --code tomsk --name Томск --cases "Заб Томск.xlsx" --pcr "ПЦР Томск.xlsx"
"""
import argparse

from ai4epi.data.raw_regions import RAW_CITIES, convert_raw_city


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", help="Каталог с исходными xlsx известных городов.")
    parser.add_argument("--city", action="append", choices=sorted(RAW_CITIES))
    parser.add_argument("--code", help="Код нового города (имя каталога в data/regions).")
    parser.add_argument("--name", help="Название города по-русски (для отчёта).")
    parser.add_argument("--district", help="Название города в общероссийской выгрузке ПЦР (по умолчанию --name).")
    parser.add_argument("--cases", help="Файл «Заб <город>.xlsx».")
    parser.add_argument("--pcr", help="Файл «ПЦР <город>.xlsx».")
    parser.add_argument("--out", default="data/regions")
    parser.add_argument("--start-year", type=int, default=2010)
    args = parser.parse_args()
    if args.code:
        if not (args.cases and args.pcr):
            parser.error("для --code нужны --cases и --pcr")
        print(convert_raw_city(args.code, out_dir=args.out, start_year=args.start_year, cases_file=args.cases,
                               pcr_file=args.pcr, name_ru=args.name, district=args.district))
        return
    if not args.raw_dir:
        parser.error("нужен --raw-dir или --code")
    for city in args.city or [c for c in RAW_CITIES if c != "spb_raw"]:
        print(convert_raw_city(city, raw_dir=args.raw_dir, out_dir=args.out, start_year=args.start_year))


if __name__ == "__main__":
    main()
