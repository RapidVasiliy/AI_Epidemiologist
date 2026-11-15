# Неопределённость матрицы контактов в возрастной модели Барояна–Рвачёва

Матрица 0–14 / 15+ (`RSCF_CONTACT_MATRIX_2`) получена агрегированием матрицы 4×4 (0–2, 3–6, 7–14, 15+) из
RSCF_Uncertainty, которая, в свою очередь, оценена косвенно по Ajelli & Litvinova (2017, J. Theor. Biol. 419:1–7).
Здесь проверяется, насколько выводы возрастной модели зависят от погрешности элементов матрицы.

| Файл | Что делает |
|---|---|
| `run.py` | 5 дат (СПб 21.01.2019, 11.03.2019, 29.01.2024, 12.02.2018; Челябинск 16.01.2023) × матрицы: базовая, исправленная на взаимность, по 40 случайных с логнормальной погрешностью каждого элемента σ = 0.25 и 0.5; калибровка — отжиг, один seed |
| `analyze.py` | сводка по `run.py` → `results/single_seed_summary.txt` |
| `seeds.py` | разброс из-за одного лишь seed оптимизатора (базовая матрица, 10 seed) |
| `best4.py`, `an2.py` | две даты фазы роста/пика, для каждой матрицы — лучшая из 4 калибровок (шум оптимизатора убран), 30 матриц на σ → `results/best_of_4_seeds_summary.txt` |

```bash
python research/contact_matrix_uncertainty/run.py out.jsonl && python research/contact_matrix_uncertainty/analyze.py out.jsonl
python research/contact_matrix_uncertainty/best4.py research/contact_matrix_uncertainty best4.jsonl && python research/contact_matrix_uncertainty/an2.py best4.jsonl
```

Выводы — в `docs/age_groups.md`, раздел «Неопределённость матрицы контактов».
