"""Проверки функций news_eda на искусственных данных.

run_code_checks() выполняет их и возвращает названия пройденных проверок; ноутбук показывает их в приложении.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac, cov_nw_groupsum

from news_eda import (
    ID_TOLERANCE_DAYS, LATE_OPEN_MINUTE, SESSION_START, TZ, audit_document_integrity, bootstrap_loss_comparison,
    clean_news_mask, forward_windows, id_dates, main_session_bars, next_session, paired_date_losses,
    session_hac_covariance, url_dates,
)


def check_next_session():
    fixture_dates = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"])
    pub = pd.Series(
        pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-07", None]), index=[8, 3, 2, 1]
    )
    mapped = next_session(pub, fixture_dates)
    assert mapped.loc[8] == pd.Timestamp("2025-01-06")
    assert mapped.loc[3] == pd.Timestamp("2025-01-06")
    assert mapped.loc[[2, 1]].isna().all()


def check_forward_windows():
    fixture = pd.DataFrame(
        {
            "instrument_id": ["TEST"] * 6,
            "trade_date": pd.bdate_range("2025-01-06", periods=6),
            "open": [100.0] * 6,
            "close": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "valid_bar": [True] * 6,
            "dividend": [False] * 6,
            "scale_break": [False] * 6,
        }
    )
    trial = forward_windows(fixture)
    assert np.isclose(trial.raw_y_3.iloc[0], 0.03)
    assert pd.isna(trial.raw_y_3.iloc[-1])
    fixture.loc[1, "valid_bar"] = False
    assert pd.isna(forward_windows(fixture).raw_y_3.iloc[0])
    fixture.loc[1, "valid_bar"] = True
    fixture.loc[2, "scale_break"] = True
    assert pd.isna(forward_windows(fixture).raw_y_3.iloc[0])
    fixture.loc[2, "scale_break"] = False
    fixture.loc[2, "dividend"] = True
    assert forward_windows(fixture).div_window_3.iloc[0]


def check_main_session_bars():
    # Основная сессия: свечи 06:59 и 19:05 не входят, порядок строк не важен; без свечи 10:00 открытие позднее.
    minute_fixture = pd.DataFrame({
        "instrument_id": "TEST", "day": pd.Timestamp("2025-01-06"), "minute": [-181, 0, 7, 529, 545],
        "open": [90.0, 100.0, 101.0, 103.0, 110.0], "high": [95.0, 102.0, 104.0, 103.5, 120.0],
        "low": [89.0, 99.0, 100.0, 102.5, 109.0], "close": [94.0, 101.0, 102.0, 103.0, 115.0],
    })
    minute_fixture["bar_ts"] = (minute_fixture.day + pd.to_timedelta(SESSION_START + minute_fixture.minute, unit="min")).dt.tz_localize(TZ)
    main_fixture = main_session_bars(minute_fixture.iloc[::-1]).iloc[0]
    assert (main_fixture.open, main_fixture.high, main_fixture.low, main_fixture.close) == (100.0, 104.0, 99.0, 103.0)
    assert main_fixture.open_minute == 0 and main_fixture.last_bar == minute_fixture.bar_ts[3]
    late_fixture = main_session_bars(minute_fixture.loc[minute_fixture.minute.ne(0)]).iloc[0]
    assert late_fixture.open == 101.0 and late_fixture.open_minute > LATE_OPEN_MINUTE


def check_document_audit():
    audit_fixture = pd.DataFrame({
        "title": ["Компания объявила новый производственный план", "Компания объявила новый производственный план",
                  "Пользователи сообщили о сбое в Instagram", "Краткая заметка", "Заметка", "Заметка", "Заметка"],
        "text": [
            "Компания объявила новый производственный план\nМосква, 11 мая, 2024, 12:00 — ИА Регнум. Текст.",
            "Компания объявила новый производственный план\nМосква, 28 апреля, 2023, 12:00 — ИА Регнум. Текст.",
            "Украинский экономист объяснил падение гривны\nКиев, 6 сентября, 2012, 11:24 — ИА Регнум. Текст.",
            "Текст без независимой даты.", "Текст без датировки.", "Текст без датировки.", "Текст без датировки.",
        ],
        "published_at": pd.to_datetime(
            ["2024-05-11T09:00Z"] * 4 + ["2024-05-10T23:59Z", "2024-05-11T09:00Z", "2024-05-09T09:00Z"]
        ),
        "url": ["https://regnum.ru/news/1"] * 4 + [
            "https://lenta.ru/news/2024/05/10/example/", "https://lenta.ru/news/2020/05/10/example/",
            "https://lenta.ru/news/2024/05/10/example/",
        ],
    })
    audit_trial = audit_document_integrity(audit_fixture)
    assert not audit_trial.integrity_quarantine.iloc[0]
    assert audit_trial.body_date_conflict.iloc[1]
    assert audit_trial.title_body_conflict.iloc[2] and audit_trial.body_date_conflict.iloc[2]
    assert not audit_trial.date_checked.iloc[3] and not audit_trial.integrity_quarantine.iloc[3]
    assert not audit_trial.url_date_conflict.iloc[4]  # Один день из-за московской даты допустим.
    assert audit_trial.url_date_conflict.iloc[5]
    assert audit_trial.url_date_conflict.iloc[6]  # published_at раньше даты в URL.


def check_url_and_id_dates():
    url_fixture = pd.Series(["https://ria.ru/20240625/x-1.html", "https://www.rbc.ru/finances/01/03/2024/abc",
                             "https://lenta.ru/news/2024/05/10/x/", "https://www.kommersant.ru/doc/7248149"])
    assert url_dates(url_fixture).tolist()[:3] == list(pd.to_datetime(["2024-06-25", "2024-03-01", "2024-05-10"]))
    assert pd.isna(url_dates(url_fixture).iloc[3])
    # 60 статей по одной в день и старая статья (номер начала января) с датой 25.02.2024.
    id_published = pd.Series(list(pd.date_range("2024-01-01", periods=60)) + [pd.Timestamp("2024-02-25")])
    id_urls = pd.Series([f"https://tass.ru/ekonomika/{1_000_000 + 1_000 * i}" for i in range(60)]
                        + ["https://tass.ru/ekonomika/1005500"])
    id_fixture = id_dates(id_urls, pd.Series(pd.NaT, index=id_urls.index), id_published)
    assert (id_fixture.iloc[:60] == id_published.iloc[:60]).all()
    assert abs((id_published.iloc[60] - id_fixture.iloc[60]).days) > ID_TOLERANCE_DAYS


def check_repeat_filter():
    repeat_fixture = pd.DataFrame({
        "doc_id": ["bad-first", "valid-copy", "other-company", "low-relevance-first"],
        "instrument_id": ["MOEX:SBER", "MOEX:SBER", "MOEX:GAZP", "MOEX:SBER"],
        "hash_key": ["same"] * 4, "max_relevance": [1.0, 0.7, 1.0, 0.3],
        "lang": ["ru"] * 4, "has_body": [True] * 4, "lang_bad": [False] * 4,
        "date_bad": [False] * 4, "recap": [False] * 4,
        "integrity_quarantine": [True, False, False, False],
        "publication_date": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-03", "2024-01-01"]),
        "published_at": pd.to_datetime(["2024-01-02T00:00Z", "2024-01-04T00:00Z", "2024-01-03T00:00Z", "2024-01-01T00:00Z"]),
    })
    assert clean_news_mask(repeat_fixture).tolist() == [False, True, True, False]


def check_hac_grid():
    hac_dates = pd.bdate_range("2024-01-01", periods=5)
    hac_observed = hac_dates[[0, 1, 3, 4]]
    hac_x = np.ones((4, 1))
    hac_residual = np.array([1.0, -1.0, 1.0, -1.0])
    hac_gap = session_hac_covariance(hac_x, hac_residual, hac_observed, hac_dates, maxlags=1, correction=False)
    assert np.isclose(hac_gap[0, 0], 0.125)  # Две соседние пары; пропуск не создаёт третью.
    hac_compressed = cov_hac(sm.OLS(hac_residual, hac_x).fit(), nlags=1, use_correction=False)
    assert np.isclose(hac_compressed[0, 0], 0.0625)
    rng_fixture = np.random.default_rng(123)
    x_fixture = sm.add_constant(rng_fixture.normal(size=(80, 2)))
    fit_fixture = sm.OLS(rng_fixture.normal(size=80), x_fixture).fit()
    continuous_dates = pd.bdate_range("2024-01-01", periods=80)
    np.testing.assert_allclose(
        session_hac_covariance(x_fixture, fit_fixture.resid, continuous_dates, continuous_dates, maxlags=10),
        cov_hac(fit_fixture, nlags=10, use_correction=True), rtol=1e-10, atol=1e-12,
    )


def check_hac_panel():
    rng_fixture = np.random.default_rng(123)
    continuous_dates = pd.bdate_range("2024-01-01", periods=80)
    x_panel = sm.add_constant(rng_fixture.normal(size=(160, 2)))
    fit_panel = sm.OLS(rng_fixture.normal(size=160), x_panel).fit()
    np.testing.assert_allclose(  # две бумаги на одних датах: суммы по сессиям, как у Driscoll–Kraay
        session_hac_covariance(x_panel, fit_panel.resid, np.tile(continuous_dates, 2), continuous_dates, maxlags=10),
        cov_nw_groupsum(fit_panel, 10, np.tile(np.arange(80), 2), use_correction="hac"), rtol=1e-10, atol=1e-12,
    )


def check_paired_bootstrap():
    prediction_fixture_rows = []
    prediction_fixture_dates = pd.bdate_range("2025-01-01", periods=16)
    for day, date in enumerate(prediction_fixture_dates):
        for asset in range(2):
            for model, prediction in [("Ноль", 0.0), ("Цена", 0.001), ("Цена + новости", 0.001)]:
                prediction_fixture_rows.append(dict(row_id=2 * day + asset, fold=f"Q{day // 8 + 1}", date=date,
                                                    ticker=str(asset), y=0.01 * (-1) ** day, model=model, prediction=prediction))
    prediction_fixture = pd.DataFrame(prediction_fixture_rows)
    bootstrap_fixture = bootstrap_loss_comparison(prediction_fixture, prediction_fixture_dates, block_size=3, draws=100)
    assert bootstrap_fixture[["improvement_pct", "low_pct", "high_pct"]].eq(0).all().all()
    sparse_fixture = prediction_fixture.loc[prediction_fixture.date.ne(prediction_fixture_dates[6])]
    sparse_daily = paired_date_losses(sparse_fixture, prediction_fixture_dates)
    assert len(sparse_daily) == len(prediction_fixture_dates)
    assert sparse_daily.loc[sparse_daily.date.eq(prediction_fixture_dates[6]), "N"].eq(0).all()
    try:
        paired_date_losses(prediction_fixture.iloc[1:], prediction_fixture_dates)
    except ValueError:
        pass
    else:
        raise AssertionError("Bootstrap должен отвергать несовпадающие наборы строк")


CODE_CHECKS = OrderedDict([
    ("Выходные, null и правая граница привязки", check_next_session),
    ("Полное будущее окно, разрыв шкалы и дивиденды", check_forward_windows),
    ("Основная сессия: окно 10:00–18:50, порядок свечей и позднее открытие", check_main_session_bars),
    ("Карантин дат, заголовков, URL, ранней published_at и неизвестная дата", check_document_audit),
    ("Дата из URL РИА/РБК и по номеру статьи", check_url_and_id_dates),
    ("Непригодный дубль не подавляет пригодный; дедупликация по акции", check_repeat_filter),
    ("HAC сохраняет пропуск и совпадает со statsmodels на полной сетке", check_hac_grid),
    ("HAC по панели совпадает с Driscoll–Kraay из statsmodels", check_hac_panel),
    ("Парный bootstrap: одинаковые прогнозы, сетка с пропуском и непарные строки", check_paired_bootstrap),
])


def run_code_checks():
    """Выполнить все проверки по порядку; первая неудача останавливает расчёт (AssertionError)."""
    passed = OrderedDict()
    for label, check in CODE_CHECKS.items():
        check()
        passed[label] = True
    return passed
