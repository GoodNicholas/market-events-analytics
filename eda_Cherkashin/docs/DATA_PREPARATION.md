# Данные для EDA

Локальная выборка — `data/ru_eda_2024_2026/` в корне репозитория, в Git не входит. S3 используется только для чтения.
Инструменты: SBER, GAZP, LKOH, GMKN, NVTK, PLZL, ROSN, VTBR и IMOEX.
Расчёты — [ноутбук](../01_financial_news_stocks_eda.ipynb), результаты — [README.md](../README.md).

## Состав

| Данные | Источник | Файлы |
|---|---|---|
| Новости и связи с бумагами | S3 | `docs.parquet`, `news_instruments.parquet`, `doc_flags.parquet` |
| Минутные свечи MOEX, 2024–2026 | S3 | `quotes_1m.parquet` |
| Минутные свечи MOEX, 2018–2023 | MOEX ISS | `data/ru_quotes_2018_2023/quotes_1m.parquet` |
| Дневные свечи и дивиденды | S3 | `daily/` |
| Календарь торгов, корпоративные действия | S3 | `reference/` |
| Тональность, события, ключевая ставка | S3 | `extras/doc_sentiment.parquet`, `extras/events.parquet`, `extras/macro_series.parquet` |
| Отраслевые индексы | MOEX ISS, копия из `Optimization_experiments` | `extras/indices.parquet`, `extras/sector_indices.parquet` |
| Посты MarketTwits с точным временем | копия из `Optimization_experiments` | `markettwits_web.jsonl` |
| GPT-метки знака постов о нефтегазе | копия из `Optimization_experiments/labeling` | `extras/labels_gpt_v2.csv` |

Остальные файлы каталога ноутбук не читает. Покрытие и качество выгрузки — в `DATASET_REPORT.md` и `quality_report.json` там же.

## Ограничения

- Новости отобраны по готовым связям с компаниями, без порога релевантности; общерыночных новостей нет.
- Минутные свечи пересчитаны на сплиты задним числом (GMKN 08.04.2024, PLZL 27.03.2025, VTBR 15.07.2024), дневные — нет.
- Цены не скорректированы на дивиденды.
- В новостях пробел за май–сентябрь 2026 года, поэтому EDA использует 2024–2025.
