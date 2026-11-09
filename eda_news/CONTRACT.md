# Контракт артефактов витрины `notebooks/EDA_report.ipynb`

Ноутбук ничего не считает по озеру: он читает готовые файлы из каталога артефактов и подписывает под
каждым рисунком файл-источник, число строк, дату и commit. Этот документ — единственное место, где
описано, какие файлы и с какими колонками должен отдать экспорт (`mea export-report`).

## Общие правила

* Каталог: `data/cache/report/` (переменная `MEA_REPORT_DIR`). Один артефакт — один файл: `<name>.parquet`
  или `<name>.csv` для таблиц (ноутбук пробует parquet, затем csv), `<name>.json` для словарей/списков.
* Тот же набор отдаёт сервер: `GET {MEA_REPORT_API}/report/artifacts/<name>.<ext>` (бэкенд `api`).
* `manifest.json` обязателен: `generated_at`, `git_commit` (`git describe --dirty`), `dev_end`, `demo: false`,
  `artifacts: {name: {file, rows, mtime, producer}}` — `producer` = модуль/скрипт, который посчитал числа.
* Все числа — за период разработки (до `dev_end` включительно); экспорт обязан идти через загрузчики с
  `assert_dev_only`. Холдаут в артефактах не появляется ни в каком виде.
* Доли — в долях 0…1 (ноутбук сам рисует проценты); «п.п.» — только там, где так сказано в описании.
* Интервалы — колонки `<x>_lo` / `<x>_hi` (95 %, тем же методом, что в MEA_RESULTS.md: Уилсон для долей,
  бутстрап по датам для событийных величин и т. д.). Если интервал не считался — колонки есть, значения пустые.
* Строки реестра: там, где артефакт — результат гипотезы, в нём есть `registry_id` (стабильный `id` строки
  реестра), чтобы число в витрине можно было найти в реестре.
* Текстовые колонки — по-русски, как в MEA_RESULTS.md (`market` — `RU`/`US`; язык холдаута — `RU`/`EN`).
* Отсутствующий артефакт — не ошибка: ноутбук показывает заглушку «артефакт не найден» и продолжает.
  Приоритет P1 — без него витрина для защиты неполна; P2 — желательно; P3 — украшение.
* Большие таблицы (`c1_points`) — parquet; остальное можно csv. Размер каталога целиком — не больше ~50 МБ.

## Артефакты

| раздел | артефакт | формат | приоритет | колонки / содержимое |
|---|---|---|---|---|
| 0 | `manifest` | json | P1 | generated_at, git_commit, dev_end, demo(bool), artifacts{name:{file,rows,mtime,producer}} |
| 1.1 | `layers` | csv | P1 | слой конвейера: layer, rows, period_from, period_to, note (порядок = порядок конвейера) |
| 1.1 | `docs_monthly` | csv | P1 | month(YYYY-MM), market(RU/US), source_class(пресса/раскрытия/агрегаторы/прочее), docs |
| 1.1 | `docs_by_source` | csv | P2 | market, source, docs, share, first_date, last_date — топ-20 источников на рынок |
| 1.2 | `dedup` | json | P2 | docs_total, after_exact, after_near, clusters, mega_cluster_empty_sha, minhash{perm,shingle,jaccard,window_days} |
| 1.3 | `attribution_funnel` | json | P1 | {RU:{stages:[{name,error}], holdout:{id,n,error,lo,hi,strict_precision,loss_true}}, EN:{…}} — ошибки в долях |
| 1.4 | `time_precision` | csv | P1 | market(RU/US/CN/EU/OTHER), source_class(источник: lenta, ccnews, fnspid…), precision(минута/только дата), events |
| 1.5 | `seam_returns` | csv | P2 | basis(close/adj_close), where(внутри источника/шов), median_abs_r, p99, share_gt_25 |
| 1.5 | `price_layer` | json | P1 | excluded_tapes, excluded_tickers, cut_tapes, dropped_days, event_days_total, event_days_no_window, unexplained_total, unexplained_share, unexplained_in_window, detector('|log r|>2') |
| 1.5 | `price_jumps_by_year` | csv | P2 | tape, year, jumps, explained |
| 1.6 | `targets` | json | P2 | RU, US (число целей), holdout_skipped{RU,US}, beta_window[-250,-30], min_obs, horizons |
| 2.1 | `facts_ch01_04` | csv | P1 | fact, method, ru, ru_lo, ru_hi, us, us_lo, us_hi, unit(share/count/slope), key(bool — для дамббелла) |
| 2.1 | `zipf` | csv | P2 | market, rank, freq (ранги 1…10 000, можно прорежённые), + slope, slope_lo, slope_hi в каждой строке |
| 2.1 | `heaps` | csv | P3 | market, tokens, vocab, beta, beta_lo, beta_hi |
| 2.1 | `doc_length` | csv | P2 | market, field(title/text), bin_lo, bin_hi, docs — гистограмма длины в токенах |
| 2.1 | `sentiment_dist` | csv | P1 | market(RU/US/все), instrument(модель), model_name, bin_lo, bin_hi, docs — гистограмма тональности [-1,1]; словарного балла в озере нет |
| 2.1 | `sentiment_agreement` | csv | P1 | lang(ru/en/de/…), model_sign(neg/neu/pos), dict_sign, docs |
| 2.1 | `event_types` | csv | P1 | market, type, events, share |
| 2.1 | `mentions_lorenz` | csv | P2 | market, tickers_cum_share, mentions_cum_share (кривая Лоренца), + gini, silent_tickers, tickers_total |
| 2.2 | `tails` | csv | P1 | series, window, n, kurt, kurt_lo, kurt_hi, hill_r, hill_r_lo, hill_r_hi, hill_l, hill_l_lo, hill_l_hi, market |
| 2.2 | `kurtosis_sensitivity` | json | P2 | series, full, without_worst_day, worst_day, worst_day_logret |
| 2.2 | `aggregation_gaussianity` | csv | P2 | venue, horizon(1m/5m/30m), kurt_full, kurt_wo_worst |
| 2.3 | `acf` | csv | P1 | series, market, lag(1…40), acf_r, acf_absr, band (1.96/sqrt(n)) |
| 2.3 | `longmemory` | csv | P1 | series, window, acf_r1, lo, hi, acf_absr1, lo, hi, hurst_absr, lo, hi, hurst_r_control, gph_d, lo, hi, leverage_k1, lo, hi |
| 2.3 | `minute_acf` | csv | P2 | venue, zero_share, acf1, band |
| 2.4 | `volume_vol` | csv | P1 | series, corr, corr_lo, corr_hi, g_vol2vola, lo, hi, g_vola2vol, lo, hi, robust_specs(e.g. '12/12') |
| 2.4 | `gainloss_asym` | csv | P2 | series, asym, asym_lo, asym_hi, ratio_5, ratio_10, ratio_20 |
| 2.5 | `intraday_profile` | csv | P1 | venue(MOEX/US), time(HH:MM по зоне биржи), absr_mult, absr_lo, absr_hi, vol_mult, vol_lo, vol_hi |
| 2.5 | `intraday_sessions` | csv | P1 | venue, measure(|r|/объём), session, mult, lo, hi, t_from, t_to |
| 2.5 | `open_mult_by_year` | csv | P2 | venue, year, mult, lo, hi |
| 2.6 | `weekday` | csv | P1 | market, weekday(пн…пт), mult, lo, hi |
| 2.6 | `calendar_effects` | csv | P1 | market, effect, absr_ratio, absr_lo, absr_hi, turn_ratio, turn_lo, turn_hi |
| 2.6 | `exdate` | csv | P1 | market, events, dates, med_div, med_ret, resid_med, resid_lo, resid_hi, ratio_sums, ratio_lo, ratio_hi, share_less, dropped |
| 2.6 | `exdate_dist` | csv | P3 | market, bin_lo, bin_hi, events — гистограмма остатка r + div/P(-1), п.п. |
| 3.2 | `ch06_contrast` | csv | P1 | measure(размах/оборот), market, pairs, dates, c0, c0_lo, c0_hi, c1, c1_lo, c1_hi, c2, c2_lo, c2_hi, pl, pl_lo, pl_hi |
| 3.2 | `ch06_dose` | csv | P1 | market, n_events(1,2,3,4-5,6-9,10+), diff, lo, hi, days |
| 3.2 | `ch06_newsdays_by_year` | csv | P1 | market, year, share_news_days |
| 3.2 | `ch06_bigmove` | csv | P2 | market, share_news_in_bigmove, lo, hi, base_share |
| 3.3 | `ch06_timeknown` | csv | P1 | market, subsample(настоящее время/полночь), news_days, tickers, absr, absr_lo, absr_hi, turn, turn_lo, turn_hi |
| 3.3 | `event_window_profile` | csv | P1 | market, subsample(все/настоящее время/плацебо), k(-10…10), absr_n, absr_lo, absr_hi, turn_r, turn_lo, turn_hi |
| 4 | `hypotheses` | csv | P1 | id, market, family, variant, metric, unit, estimate, lo, hi, t, n, placebo, placebo_lo, placebo_hi, verdict(подтверждена/отличима от нуля, ниже порога/не различима/опровергнута/недостаточно наблюдений/в работе), note, registry_id |
| 4.2 | `h1b_diagnostics` | csv | P1 | run, t, gain, lo, hi, qlike_pct (доля от потери HAR, в %), order |
| 4.4 | `h15_items` | csv | P1 | item, label, n, abs_ar, vol_ratio, diff_vs_202, lo, hi |
| 4.5 | `h16_horizons` | csv | P1 | horizon_min, vol_share, vol_lo, vol_hi, move_share, move_lo, move_hi, n |
| 4.5 | `h16_gap` | json | P1 | ratio_sums, lo, hi, median, med_lo, med_hi, n, mean_of_ratios_rejected |
| 4.6 | `h19_quintiles` | csv | P1 | quintile, label, events, dates, drift, lo, hi, t, p, + diff_1_5, diff_lo, diff_hi, size_proxy |
| 4.3 | `h3_terciles` | csv | P2 | market, tercile(неликвид/середина/ликвид), est, lo, hi, n |
| 4.3 | `h13_tiers` | csv | P2 | market, tier(30/50/70/100), est, lo, hi, n, + slope, slope_lo, slope_hi, topbottom, tb_lo, tb_hi |
| 5.1 | `c1_points` | parquet | P1 | doc_id, x, y, cluster(-1 = шум), market, year, source_class, ticker (первый тикер яруса 100), sentiment |
| 5.1 | `c1_clusters` | csv | P1 | cluster, n, label (имя/топ-слова), share_ru, absr, absr_lo, absr_hi, turn, turn_lo, turn_hi, cx, cy, top_tickers, top_sources |
| 5.1 | `c1_titles` | csv | P2 | doc_id, cluster, title, date, source, ticker — по 5–10 типичных заголовков на кластер (ближайшие к центроиду) |
| 5.1 | `c1_cluster_monthly` | csv | P2 | cluster, month, docs — для 8–10 выделенных кластеров |
| 5.1 | `c1_cluster_types` | csv | P3 | cluster, type, share — состав по размеченным типам событий |
| 5.1 | `c1_params` | json | P2 | embedding_model, dim, pca, umap{n_neighbors,min_dist}, hdbscan{min_cluster_size}, sample_n, clusters, noise_share |
| 5.2 | `c2_results` | csv | P1 | market, horizon(nowcast/forecast), task(AUC/R2), p, pn, gain, lo, hi, placebo, n_events, n_test, test_from, passed(bool) |
| 5.2 | `c2_importance` | csv | P1 | market, horizon, task, method(gain/permutation), prices, news |
| 5.3 | `c3_summary` | json | P1 | macro_f1, baseline_majority, baseline_stratified, split_date, n_train, n_test, model |
| 5.3 | `c3_per_class` | csv | P1 | item, label, support, precision, recall, f1 |
| 5.3 | `c3_confusion` | csv | P1 | true, pred, count |
| 5.3 | `c3_funnel` | csv | P2 | step, count |
| 5.4 | `case_prices` | csv | P1 | ticker, date, close(сырая), adj_close, volume — одна бумага (по умолчанию SBER), 2015…dev_end |
| 5.4 | `case_events` | csv | P1 | ticker, date, time_known(bool), source, source_class, title, sent_model, sent_dict, tier, type, cluster |
| 5.4 | `case_reaction` | csv | P2 | ticker, subsample(события/плацебо), k, absr_n, lo, hi, turn_r, turn_lo, turn_hi |
| 5.4 | `case_neighbors` | csv | P2 | query_title, query_date, nb_title, nb_date, nb_source, nb_ticker, cos — 3 запроса × 5 соседей (bge-m3) |
| 6 | `controls` | csv | P1 | control, how, where |
| 6 | `registry_stats` | json | P1 | rows, live, voided, by_kind{}, by_family{}, hosts, first_date, last_date |
| 6 | `registry_timeline` | csv | P2 | date, added, voided |
| 7 | `not_measured` | csv | P1 | item, why |
| 8 | `statements` | json | P1 | [{n, title, text, hero(value, label)}] — семь утверждений + утверждение о данных |

## Соответствие разделам MEA_RESULTS.md

| раздел ноутбука | раздел отчёта | артефакты |
|---|---|---|
| 0 | шапка, карта артефактов | `manifest` |
| 1.1 | 1.1 конвейер | `layers`, `docs_monthly`, `docs_by_source` |
| 1.2 | 1.2 дедупликация | `dedup` |
| 1.3 | 1.3 привязка | `attribution_funnel` |
| 1.4 | 1.4 safe bar | `time_precision` |
| 1.5 | 1.5 ценовой слой | `seam_returns`, `price_layer`, `price_jumps_by_year` |
| 1.6 | 1.6 цели | `targets` |
| 2.1 | 2.1 главы 01–04 + корпус как текст, тональность, типы | `facts_ch01_04`, `zipf`, `heaps`, `doc_length`, `sentiment_dist`, `sentiment_agreement`, `event_types`, `mentions_lorenz` |
| 2.2 | 2.2 хвосты | `tails`, `kurtosis_sensitivity`, `aggregation_gaussianity` |
| 2.3 | 2.3 автокорреляция, долгая память | `acf`, `longmemory`, `minute_acf` |
| 2.4 | 2.4 объём/волатильность, асимметрия | `volume_vol`, `gainloss_asym` |
| 2.5 | 2.5 внутридневная сезонность | `intraday_profile`, `intraday_sessions`, `open_mult_by_year` |
| 2.6 | 2.6 календарь, ex-date | `weekday`, `calendar_effects`, `exdate`, `exdate_dist` |
| 3.2 | 3.1–3.2 глава 06: контрасты, доза | `ch06_contrast`, `ch06_dose`, `ch06_newsdays_by_year`, `ch06_bigmove` |
| 3.3 | 3.3 реакция/ассоциация, окно −10…+10 | `ch06_timeknown`, `event_window_profile` |
| 4 | 4 сводная таблица гипотез | `hypotheses` |
| 4.2 | 4.2 H1/H1b | `h1b_diagnostics` |
| 4.4 | 4.4 H15 | `h15_items` |
| 4.5 | 4.5 H16 | `h16_horizons`, `h16_gap` |
| 4.6 | 4.6 H19 | `h19_quintiles` |
| 4.3 | 4.3 H3/H13 | `h3_terciles`, `h13_tiers` |
| 5.1 | 5.1 C1 карта сюжетов | `c1_points`, `c1_clusters`, `c1_titles`, `c1_cluster_monthly`, `c1_cluster_types`, `c1_params` |
| 5.2 | 5.2 C2 | `c2_results`, `c2_importance` |
| 5.3 | 5.3 C3 | `c3_summary`, `c3_per_class`, `c3_confusion`, `c3_funnel` |
| 5.4 | 5.4 одна бумага крупным планом (новое) | `case_prices`, `case_events`, `case_reaction`, `case_neighbors` |
| 6 | 6 контроль | `controls`, `registry_stats`, `registry_timeline` |
| 7 | 7 не измерено | `not_measured` |
| 8 | 8 семь утверждений | `statements` |

## Как проверить экспорт

```
mea export-report --out data/cache/report      # пишет все артефакты + manifest.json
python notebooks/run_check.py --strict --backend local --html reports/EDA_report_check.html
jupyter nbconvert --to html --execute --no-input notebooks/EDA_report.ipynb --output ../reports/EDA_report.html
```

`run_check.py` исполняет ячейки без Jupyter и печатает, какие артефакты не найдены и где упало построение.
Карта артефактов в начале ноутбука показывает то же самое в самом отчёте.
