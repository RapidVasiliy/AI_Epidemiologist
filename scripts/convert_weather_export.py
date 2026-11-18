"""Выгрузки погоды (архив rp5.ru или любая таблица с датой и температурой) → недельная таблица.

python scripts/convert_weather_export.py "SPb 2010-2014.xls" "SPb 2015-2024.xls" --out data/weather/spb_weekly.csv
ai4epi run-analysis-local --city spb --weather-weekly data/weather/spb_weekly.csv ...

Подробности форматов — ai4epi.data.weather_import.
"""
import argparse
from pathlib import Path

from ai4epi.data.weather_import import convert_weather_exports, fill_short_gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", nargs="+", type=Path, help="Файлы выгрузки (xls/xlsx/csv), можно несколько.")
    parser.add_argument("--out", required=True, type=Path, help="Недельная таблица CSV.")
    parser.add_argument("--min-observations", type=int, default=4, help="Минимум наблюдений в неделе.")
    parser.add_argument("--max-gap-weeks", type=int, default=3, help="Заполнять интерполяцией разрывы не длиннее.")
    args = parser.parse_args()
    weekly = convert_weather_exports(args.exports, min_observations=args.min_observations)
    weekly, long_gaps = fill_short_gaps(weekly, max_gap_weeks=args.max_gap_weeks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.out, index=False)
    print(f"{args.out}: {len(weekly)} недель, {weekly['week_start'].min().date()} – {weekly['week_start'].max().date()}, "
          f"влажность: {'есть' if 'rh_mean' in weekly else 'нет'}")
    if long_gaps:
        print("Длинные разрывы (анализ на этих неделях упадёт — нужна дополнительная выгрузка):")
        for first, last in long_gaps:
            print(f"  {first} – {last}")


if __name__ == "__main__":
    main()
