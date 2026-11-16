# Сравнение ядра модели (БР / SEIR) и сэмплера (MCMC / NUTS) на бэктесте

Итоги и выводы — `docs/mechanistic_wave.md`, раздел «Сравнение на бэктесте: ядро × сэмплер».

```bash
# один вариант, сезоны из файла (город, год начала сезона)
bash research/kernel_comparison/run.sh seir mcmc 2        # → out_seir_mcmc/<город>_<год>/predictions.csv
python research/kernel_comparison/compare.py seasons_nuts.txt seir_mcmc br_nuts seir_nuts
```

`run.sh` вызывает `ai4epi backtest-local --br-kernel ... --br-method ...` для каждого сезона;
`compare.py` сравнивает варианты на общих точках волны (MAE по фазам и горизонтам, R², R-hat, покрытие
конформной полосы с исключением сезона). Пути к каталогам результатов в скриптах — локальные, их нужно
поправить под свою машину. Вариант «БР + MCMC» взят из бэктестов, по которым построена
`data/calibration/br_conformal.json`.
