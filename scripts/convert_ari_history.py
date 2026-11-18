"""Многолетняя выгрузка ОРВИ по городам (data1990_2019.xlsx) → data/regions/<город>/ari_history.csv.

    python scripts/convert_ari_history.py data1990_2019.xlsx \
        [--regions-dir data/regions] [--all-towns data/national/ari_history_51_towns.csv]

Неделя выгрузки (от недели, содержащей 1 января) переводится в понедельник ISO-недели; колонки
``ari_<g>`` (случаи), ``pop_<g>`` (население), ``epidemic_<g>`` (отметка «Э»), g = total, 0_2, 3_6, 7_14, 15.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai4epi.data.ari_history import convert_ari_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path)
    parser.add_argument("--regions-dir", type=Path, default=Path("data/regions"))
    parser.add_argument("--all-towns", type=Path, default=None, help="Сохранить все города одним CSV.")
    args = parser.parse_args()
    written = convert_ari_history(args.export, args.regions_dir, all_towns_file=args.all_towns)
    for code, path in written.items():
        t = pd.read_csv(path, parse_dates=["week_start"])
        print(
            f"{code}: {path} — {len(t)} недель, {t.week_start.min().date()} – {t.week_start.max().date()}, "
            f"эпидемических недель (total): {int(t.epidemic_total.sum())}"
        )


if __name__ == "__main__":
    main()
