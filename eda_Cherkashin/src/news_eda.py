"""Функции EDA «Новости и доходность акций РФ» — ноутбук 01_financial_news_eda.ipynb.

Здесь параметры методов и функции, которые не зависят от данных ноутбука.
Их проверки на искусственных данных — в news_eda_checks.py.
"""
from difflib import SequenceMatcher
import re

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# --- Параметры методов ---
SEED = 42
TZ = "Europe/Moscow"
BPS = 10_000
HORIZONS = (1, 3, 5)  # горизонты цели, сессий
RELEVANCE_THRESHOLD = 0.7
RELEVANCE_EPS = 1e-6
HAC_LAGS = 10
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_BLOCK = 20
FORECAST_BOOTSTRAP_DRAWS = 5_000
DATE_TOLERANCE_DAYS = 1
ID_TOLERANCE_DAYS = 30  # дата по номеру статьи грубая: ловим только крупные ошибки
TITLE_TOKEN_OVERLAP = 0.20
TITLE_SEQUENCE_SIMILARITY = 0.40
MIN_GROUP_DAYS = 10
SUPPORT_WARNING_DAYS = 40  # меньше — «мало данных»
SESSION_START = 10 * 60  # 10:00 МСК, минуты от полуночи
MAIN_LAST_MINUTE = 530  # 18:50 — конец аукциона закрытия и последнее значение IMOEX
LATE_OPEN_MINUTE = 5  # первая сделка позже 10:05: открытие не синхронно с индексом
NEWS_COUNTS = ["n_all", "n_clean"]  # число связей акция × якорная сессия: все и после отбора


# --- Вывод и общие функции ---
def show_table(frame, caption="", formats=None):
    defaults = {column: "{:,.0f}" if pd.api.types.is_integer_dtype(frame[column]) else "{:,.3f}"
                for column in frame.select_dtypes("number")}
    display(frame.style.format(defaults | (formats or {}), na_rep="—", escape="html")
            .hide(axis="index").set_caption(caption))


def show_figure(fig, title):
    fig.suptitle(title, x=0.02, ha="left")
    plt.show()
    plt.close(fig)


def assert_unique(frame, keys, name):
    """Остановить склейку, если ключи могут размножить наблюдения."""
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: повтор ключа {keys}")


def adjust_fdr(pvalues):
    """BY-коррекция только конечных p-value; неизвестные остаются NaN."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    finite = np.isfinite(pvalues)
    if finite.any():
        adjusted[finite] = multipletests(pvalues[finite], method="fdr_by")[1]
    return adjusted


def finite_mean(values):
    """Среднее конечных значений; NaN, если их нет."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return values.mean() if len(values) else np.nan


def bootstrap_by_day(frame, numerator, denominator=None, draws=BOOTSTRAP_DRAWS):
    """Отношение сумм (без denominator — среднее); 95% интервал: даты перевыбираются целиком."""
    daily = frame.groupby("day").agg(top=(numerator, "sum"), size=(numerator, "size"))
    bottom = (frame.groupby("day")[denominator].sum() if denominator else daily["size"]).to_numpy(float)
    top = daily.top.to_numpy(float)
    samples = np.random.default_rng(SEED).integers(0, len(daily), size=(draws, len(daily)))
    low, high = np.quantile(top[samples].sum(axis=1) / bottom[samples].sum(axis=1), [0.025, 0.975])
    return pd.Series({"событий": len(frame), "оценка": top.sum() / bottom.sum(), "low": low, "high": high})


# --- Проверка дат и заголовков документов ---
MONTHS_RU = {
    month: number
    for number, month in enumerate(
        "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(), start=1
    )
}
# «Москва, 11 мая, 2024, 12:00 — ИА Регнум»
DATELINE_PATTERN = (
    r"(?:^|\n)[^\n]{0,160}?(\d{1,2})\s+(" + "|".join(MONTHS_RU)
    + r"),?\s+((?:19|20)\d{2}),\s*\d{1,2}:\d{2}\s*[—–-]\s*ИА\s*Регнум"
)
# Дата в пути URL: (шаблон, позиции года, месяца и дня)
URL_DATE_PATTERNS = [
    (r"/((?:19|20)\d{2})[/-](\d{2})[/-](\d{2})(?:/|$|[-_])", (0, 1, 2)),  # /2024/05/10/
    (r"/((?:19|20)\d{2})(\d{2})(\d{2})/", (0, 1, 2)),                    # /20240510/ — РИА, Прайм
    (r"/(\d{2})/(\d{2})/((?:19|20)\d{2})/", (2, 1, 0)),                  # /10/05/2024/ — РБК
]
# Порядковый номер статьи в URL
ID_PATTERNS = {
    "regnum.ru": r"/(?:news|article)/(\d{6,})",
    "kommersant.ru": r"/doc/(\d{6,})",
    "tass.ru": r"/(\d{6,})$",
    "interfax.ru": r"/(\d{6,})$",
}


def url_path(urls):
    # Только путь: параметры запроса не считаются датой статьи.
    return urls.fillna("").str.replace(r"^https?://[^/]+", "", regex=True).str.split(r"[?#]", regex=True).str[0]


def body_dates(text):
    parts = text.str.slice(0, 1000).str.extract(DATELINE_PATTERN, flags=re.I)
    month = parts[1].str.lower().map(MONTHS_RU)
    return pd.to_datetime(dict(year=parts[2], month=month, day=parts[0]), errors="coerce")


def url_dates(urls):
    path = url_path(urls)
    result = pd.Series(pd.NaT, index=urls.index, dtype="datetime64[ns]")
    for pattern, (y, m, d) in URL_DATE_PATTERNS:
        parts = path.str.extract(pattern)
        result = result.fillna(pd.to_datetime(dict(year=parts[y], month=parts[m], day=parts[d]), errors="coerce"))
    return result


def id_dates(urls, body_date, published_date):
    """Дата по номеру статьи: монотонная связь номер → дата внутри издания (два прохода, второй без выбросов)."""
    path = url_path(urls)
    domain = urls.str.extract(r"https?://(?:www\.)?([^/]+)", expand=False)
    result = pd.Series(pd.NaT, index=urls.index, dtype="datetime64[ns]")
    for site, pattern in ID_PATTERNS.items():
        number = pd.to_numeric(path[domain.eq(site)].str.extract(pattern, expand=False), errors="coerce").dropna()
        # У Регнума большинство published_at неверны, поэтому опора — дата из текста.
        reference = (body_date if site == "regnum.ru" else published_date)[number.index].dropna()
        if len(reference) < 50:
            continue
        x = number[reference.index].to_numpy(float)
        y = reference.map(pd.Timestamp.toordinal).to_numpy(float)
        keep = np.ones(len(x), bool)
        for _ in range(2):
            model = IsotonicRegression(out_of_bounds="clip").fit(x[keep], y[keep])
            keep = np.abs(model.predict(x) - y) <= ID_TOLERANCE_DAYS
        ordinals = np.rint(model.predict(number.to_numpy(float))).astype(int)
        result[number.index] = [pd.Timestamp.fromordinal(value) for value in ordinals]
    return result


def date_conflict(published_date, independent_date):
    """published_at может отставать от независимой даты на DATE_TOLERANCE_DAYS, но не опережать её."""
    lag_days = (published_date - independent_date).dt.days
    return lag_days.lt(0) | lag_days.gt(DATE_TOLERANCE_DAYS)


def headline_tokens(value):
    return set(re.findall(r"[a-zа-яё0-9]+", str(value).lower().replace("ё", "е")))


def titles_disagree(title, heading):
    """None — заголовки слишком короткие для сравнения; True — заголовки явно разные."""
    left, right = headline_tokens(title), headline_tokens(heading)
    if min(len(left), len(right)) < 3:
        return None
    overlap = len(left & right) / len(left | right)
    similarity = SequenceMatcher(None, " ".join(sorted(left)), " ".join(sorted(right)), autojunk=False).ratio()
    return overlap < TITLE_TOKEN_OVERLAP and similarity < TITLE_SEQUENCE_SIMILARITY


def audit_document_integrity(documents):
    """Признаки согласованности документа; даты и исходные документы не меняются."""
    text = documents.text.fillna("").astype(str)
    published = pd.to_datetime(documents.published_at, utc=True, errors="coerce")
    published_date = published.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()

    result = pd.DataFrame(index=documents.index)
    result["body_date"] = body_dates(text)
    result["url_date"] = url_dates(documents.url)
    result["id_date"] = id_dates(documents.url, result.body_date, published_date)
    result["date_checked"] = result[["body_date", "url_date", "id_date"]].notna().any(axis=1)
    result["body_date_conflict"] = date_conflict(published_date, result.body_date)
    result["url_date_conflict"] = date_conflict(published_date, result.url_date)
    result["id_date_conflict"] = (published_date - result.id_date).dt.days.abs().gt(ID_TOLERANCE_DAYS)
    result["date_conflict"] = result[["body_date_conflict", "url_date_conflict", "id_date_conflict"]].any(axis=1)

    # Первая строка является заголовком только в формате «Регнума».
    result["body_headline"] = text.str.split("\n", n=1).str[0].str.strip()
    structured = result.body_date.notna() & text.str.contains("\n", regex=False)
    verdicts = pd.Series(None, index=documents.index, dtype=object)
    verdicts.loc[structured] = [
        titles_disagree(title, heading)
        for title, heading in zip(documents.title.fillna("")[structured], result.body_headline[structured])
    ]
    result["headline_checked"] = verdicts.notna()
    result["title_body_conflict"] = verdicts.eq(True)
    result["integrity_quarantine"] = result.date_conflict | result.title_body_conflict
    return result


# --- Посты MarketTwits ---
def load_markettwits(path, start, end):
    posts = pd.read_json(path, lines=True).drop_duplicates("id").dropna(subset=["text"])
    posts["date"] = pd.to_datetime(posts.date, utc=True)
    posts["day"] = posts.date.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    posts = posts.loc[posts.day.between(start, end)]
    posts["clean"] = (posts.text.str.replace(r"#\w+|https?://\S+", " ", regex=True)
                      .str.replace(r"[^\w\s.,%:;!?«»()-]", " ", regex=True)
                      .str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, 300))
    posts = posts.loc[posts.clean.str.count(r"[A-Za-zА-Яа-яЁё]") >= 15].sort_values("date")
    repost = posts.date - posts.groupby(posts.clean.str.lower()).date.shift() <= pd.Timedelta("24h")
    return posts.loc[~repost].reset_index(drop=True)


def best_post_similarity(titles, dates, posts, shift_days=0):
    """Максимальная TF-IDF-похожесть заголовка на посты с московской датой d−1..d+1 (d сдвигается на shift_days)."""
    normalize = lambda s: s.fillna("").str.lower().str.replace("ё", "е")
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w{2,}\b", ngram_range=(1, 2), min_df=2, max_df=0.01,
                                 sublinear_tf=True, dtype=np.float32)
    vectorizer.fit(pd.concat([normalize(posts.clean), normalize(titles)]))
    post_matrix = vectorizer.transform(normalize(posts.clean)).tocsr()
    title_matrix = vectorizer.transform(normalize(titles)).tocsr()
    posts_by_day = posts.groupby("day").indices
    result = np.zeros(len(titles), dtype=np.float32)
    shifted = pd.Series(dates.to_numpy(), dtype="datetime64[ns]") + pd.Timedelta(days=shift_days)
    for day, rows in shifted.groupby(shifted).indices.items():
        window = [posts_by_day.get(day + pd.Timedelta(days=k), []) for k in (-1, 0, 1)]
        candidates = np.concatenate(window).astype(int)
        if len(candidates):
            result[rows] = (title_matrix[rows] @ post_matrix[candidates].T).max(axis=1).toarray().ravel()
    return result


# --- Цены и цели ---
def main_session_bars(minute_bars):
    """Инструмент × день: OHLC по минуткам 10:00–18:50 МСК, минута первой сделки, начало последней свечи."""
    window = minute_bars.loc[minute_bars.minute.between(0, MAIN_LAST_MINUTE)].sort_values("bar_ts")
    return window.groupby(["instrument_id", "day"]).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        open_minute=("minute", "first"), last_bar=("bar_ts", "last"),
    ).reset_index()


def next_session(publication_dates, session_dates, extra_delay=0):
    """Первая сессия строго после даты; за правым краем остаётся NaT."""
    session_index = pd.DatetimeIndex(session_dates).sort_values().unique()
    publication_dates = pd.to_datetime(publication_dates)
    positions = session_index.searchsorted(publication_dates, side="right") + extra_delay
    result = pd.Series(pd.NaT, index=publication_dates.index, dtype="datetime64[ns]")
    available = publication_dates.notna().to_numpy() & (positions < len(session_index))
    result.loc[available] = session_index[positions[available]].to_numpy()
    return result


def forward_windows(frame, horizons=HORIZONS):
    """Open-to-close targets: нужны все свечи окна без внутреннего разрыва шкалы."""
    result = frame.sort_values(["instrument_id", "trade_date"]).copy()
    by_instrument = result.groupby("instrument_id", sort=False)
    for horizon in horizons:
        valid = pd.Series(True, index=result.index)
        dividend = pd.Series(False, index=result.index)
        for offset in range(horizon):
            valid &= by_instrument.valid_bar.shift(-offset, fill_value=False)
            dividend |= by_instrument.dividend.shift(-offset, fill_value=False)
            if offset:  # Разрыв до открытия якорной свечи не входит в доходность.
                valid &= ~by_instrument.scale_break.shift(-offset, fill_value=True)
        end_price = by_instrument.close.shift(-(horizon - 1))
        result[f"end_{horizon}"] = by_instrument.trade_date.shift(-(horizon - 1))
        result[f"raw_y_{horizon}"] = (end_price / result.open - 1).where(valid)
        result[f"div_window_{horizon}"] = dividend
    return result


# --- Отбор и агрегация новостей ---
def clean_news_mask(frame, relevance=RELEVANCE_THRESHOLD, *, exclude_repeats=True, policy="screened"):
    """Отбор связей; policy='metadata' — диагностический вариант без карантина."""
    relevant = frame.max_relevance.ge(relevance - RELEVANCE_EPS)
    valid_text = frame.lang.eq("ru") & frame.has_body & ~frame.lang_bad
    mask = relevant & valid_text & ~frame.date_bad & ~frame.recap
    if policy == "screened":
        mask &= ~frame.integrity_quarantine
    if exclude_repeats:
        candidates = frame.loc[mask].sort_values(["publication_date", "published_at", "doc_id"])
        duplicate = candidates.duplicated(["instrument_id", "hash_key"], keep="first")
        mask.loc[candidates.index[duplicate]] = False
    return mask


def aggregate_news(events, anchor_col="anchor", relevance=RELEVANCE_THRESHOLD,
                   policy="screened", source=None, checked_dates_only=False):
    """Отбор и агрегация; одна строка акция × якорная сессия."""
    selected = events.copy()
    if source is not None:
        selected = selected.loc[selected.source.eq(source)].copy()
    if checked_dates_only:
        selected = selected.loc[selected.date_checked & ~selected.date_conflict].copy()
    # Дедупликация до удаления правой границы: учитывается вся прошлая история отбора.
    selected["clean"] = clean_news_mask(selected, relevance, policy=policy)
    selected = selected.loc[selected[anchor_col].notna()]
    return (selected.groupby(["instrument_id", anchor_col])
            .agg(n_all=("doc_id", "size"), n_clean=("clean", "sum"))
            .reset_index().rename(columns={anchor_col: "trade_date"}))


def panel_with_news(base_panel, events, **options):
    """Изменить только наблюдаемые новостные признаки; ключи и ценовые строки неизменны."""
    counts = aggregate_news(events, **options)
    result = base_panel.drop(columns=NEWS_COUNTS).merge(
        counts, on=["instrument_id", "trade_date"], how="left", validate="one_to_one",
    )
    assert len(result) == len(base_panel)
    assert result[["instrument_id", "trade_date"]].equals(base_panel[["instrument_id", "trade_date"]])
    result[NEWS_COUNTS] = result[NEWS_COUNTS].fillna(0).astype(int)
    result["log_news"] = np.log1p(result.n_clean)
    result["has_clean_news"] = result.n_clean.gt(0).astype(float)
    return result


# --- Регрессии с HAC ---
def session_hac_covariance(exog, residuals, observation_dates, session_dates, maxlags=HAC_LAGS, correction=True):
    """Newey–West на сетке сессий: пропущенная сессия не сжимает лаг; наблюдения одной сессии (панель) суммируются."""
    x = np.asarray(exog, dtype=float)
    positions = pd.DatetimeIndex(session_dates).get_indexer(pd.DatetimeIndex(observation_dates))
    if (positions < 0).any():
        raise ValueError("Даты наблюдений отсутствуют в сетке")
    scores = np.zeros((len(session_dates), x.shape[1]))
    np.add.at(scores, positions, x * np.asarray(residuals, dtype=float)[:, None])
    meat = scores.T @ scores
    for lag in range(1, min(maxlags, len(scores) - 1) + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (maxlags + 1)) * (cross + cross.T)
    pinv = np.linalg.pinv(x)
    bread = pinv @ pinv.T
    covariance = bread @ meat @ bread
    if correction:
        covariance *= len(x) / (len(x) - np.linalg.matrix_rank(x))
    return (covariance + covariance.T) / 2


def hac_effect(y, design, dates, column, session_dates):
    """Коэффициент column в б.п. с HAC 95% интервалом и p-value; session_dates — полная сетка сессий."""
    fit = sm.OLS(y.astype(float), design).fit()
    position = design.columns.get_loc(column)
    se = np.sqrt(max(0, session_hac_covariance(design, fit.resid, dates, session_dates)[position, position]))
    df = min(fit.df_resid, pd.Index(dates).nunique() - 1)  # в панели степеней свободы не больше числа сессий
    beta, critical = fit.params[column], stats.t.ppf(0.975, df)
    return dict(beta_bps=beta * BPS, low_bps=(beta - critical * se) * BPS, high_bps=(beta + critical * se) * BPS,
                p=2 * stats.t.sf(abs(beta / se), df) if se > 0 else np.nan)


def association_design(observations, count_col, effects):
    design = pd.concat([
        observations[count_col].gt(0).astype(float).rename("news"),
        observations[["lag_r", "lag_log_value"]].astype(float),
        pd.get_dummies(observations[effects], drop_first=True, dtype=float),
    ], axis=1)
    return sm.add_constant(design, has_constant="add").astype(float)


def estimate_news_association(frame, count_col, horizon, session_dates, pooled=False):
    """Разность доходности дней с новостями и без, б.п.; pooled — все бумаги в одной регрессии с эффектом бумаги."""
    effects = ["weekday", "quarter"] + (["ticker"] if pooled else [])
    required = [f"y_{horizon}", count_col, "lag_r", "lag_log_value", "trade_date"] + effects
    observations = frame[required].dropna().sort_values("trade_date")
    exposed = observations[count_col].gt(0)
    n_on, n_off = int(exposed.sum()), int((~exposed).sum())
    by_quarter = exposed.groupby(observations.quarter).agg(["sum", "size"])
    both_in_every_quarter = (by_quarter["sum"].gt(0) & by_quarter["sum"].lt(by_quarter["size"])).all()
    result = dict(N_on=n_on, N_off=n_off,
                  limited_support=min(n_on, n_off) < SUPPORT_WARNING_DAYS or not both_in_every_quarter,
                  beta_bps=np.nan, low_bps=np.nan, high_bps=np.nan, p=np.nan)
    if len(observations) < 80 or min(n_on, n_off) < MIN_GROUP_DAYS:
        return result
    design = association_design(observations, count_col, effects)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return result
    return result | hac_effect(observations[f"y_{horizon}"], design, observations.trade_date, "news", session_dates)


def tone_difference(frame, target, session_dates, pooled=False):
    """Доходность положительных минус отрицательных новостных дней, б.п.; pooled — все бумаги с эффектом бумаги."""
    data = frame.loc[frame.n_clean.gt(0) & frame.sent_mean.ne(0), [target, "sent_mean", "trade_date", "ticker"]].dropna()
    positive = data.sent_mean.gt(0).astype(float).rename("positive")
    result = dict(N_pos=int(positive.sum()), N_neg=int((1 - positive).sum()),
                  beta_bps=np.nan, low_bps=np.nan, high_bps=np.nan, p=np.nan)
    if min(result["N_pos"], result["N_neg"]) < MIN_GROUP_DAYS:
        return result
    design = sm.add_constant(positive)
    if pooled:
        design = design.join(pd.get_dummies(data.ticker, drop_first=True, dtype=float))
    return result | hac_effect(data[target], design, data.trade_date, "positive", session_dates)


# --- Прогноз ---
def make_folds(frame, session_dates, gap=5):
    """Расширяющееся окно → квартал 2025; метки train заканчиваются до начала квартала."""
    folds = []
    for quarter in range(1, 5):
        start = pd.Timestamp(year=2025, month=3 * quarter - 2, day=1)
        safe_cut = session_dates[max(0, session_dates.searchsorted(start) - gap)]
        train = frame.trade_date.lt(safe_cut) & frame.end_5.lt(start)
        valid = frame.trade_date.ge(start) & frame.trade_date.lt(start + pd.DateOffset(months=3))
        assert set(frame.loc[train, "trade_date"]).isdisjoint(frame.loc[valid, "trade_date"])
        folds.append(dict(name=f"2025 Q{quarter}", train=frame.index[train], valid=frame.index[valid]))
    return folds


def make_model(features):
    """Масштабирование и one-hot обучаются только на train; параметры Ridge не подбираются."""
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), features),
        ("ticker", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["ticker"]),
    ])
    return make_pipeline(preprocessing, Ridge(alpha=10))


def predict_walk_forward(panel, folds, price_features, news_features):
    """Четыре модели на одних и тех же строках каждого квартала."""
    usable = pd.Series(np.isfinite(panel[price_features + news_features + ["y_1"]].to_numpy(float)).all(axis=1),
                       index=panel.index)
    models = {"Цена": price_features, "Цена + новости": price_features + news_features}
    predictions = []
    for fold in folds:
        train = panel.loc[fold["train"]].loc[usable]
        valid = panel.loc[fold["valid"]].loc[usable]
        observed = pd.DataFrame({"row_id": valid.index, "fold": fold["name"], "date": valid.trade_date.to_numpy(),
                                 "ticker": valid.ticker.to_numpy(), "y": valid.y_1.to_numpy()})
        for name, features in models.items():
            model = make_model(features).fit(train[features + ["ticker"]], train.y_1)
            predictions.append(observed.assign(model=name, prediction=model.predict(valid[features + ["ticker"]])))
        ticker_mean = valid.ticker.map(train.groupby("ticker").y_1.mean()).fillna(train.y_1.mean())
        predictions.append(observed.assign(model="Среднее по акции", prediction=ticker_mean.to_numpy()))
        predictions.append(observed.assign(model="Ноль", prediction=0.0))
    result = pd.concat(predictions, ignore_index=True)
    assert result.groupby(["row_id", "fold"]).model.nunique().eq(4).all()
    assert np.isfinite(result[["y", "prediction"]].to_numpy()).all()
    return result


def paired_date_losses(predictions, session_dates):
    """Суммы квадратов ошибок по датам на полной сетке сессий квартала; даты без строк — нули."""
    pairs = predictions.pivot(index=["row_id", "fold", "date", "y"], columns="model", values="prediction").reset_index()
    if pairs[["Цена", "Цена + новости", "Ноль"]].isna().any().any():
        raise ValueError("Наборы наблюдений моделей различаются")
    pairs = pairs.assign(price=(pairs.y - pairs["Цена"]) ** 2, news=(pairs.y - pairs["Цена + новости"]) ** 2,
                         zero=(pairs.y - pairs["Ноль"]) ** 2, N=1)
    daily = []
    for fold, data in pairs.groupby("fold", sort=True):
        grid = session_dates[(session_dates >= data.date.min()) & (session_dates <= data.date.max())]
        sums = data.groupby("date")[["price", "news", "zero", "N"]].sum().reindex(grid, fill_value=0)
        daily.append(sums.rename_axis("date").assign(fold=fold).reset_index())
    return pd.concat(daily, ignore_index=True)


def bootstrap_loss_comparison(predictions, session_dates, block_size=BOOTSTRAP_BLOCK,
                              draws=FORECAST_BOOTSTRAP_DRAWS, seed=SEED):
    """Снижение MSE от новостей, %: парный циклический блочный bootstrap дат внутри каждого квартала."""
    daily = paired_date_losses(predictions, session_dates)
    rng = np.random.default_rng(seed)

    def summarize(label, point, samples):
        sample_gain = 100 * (samples[:, 0] - samples[:, 1]) / samples[:, 0]
        low, high = np.quantile(sample_gain, [0.025, 0.975])
        return dict(fold=label, N=int(point[3]), improvement_pct=100 * (point[0] - point[1]) / point[0],
                    low_pct=low, high_pct=high, news_MSE_to_zero=point[1] / point[2])

    rows, pooled_point, pooled_draws = [], 0, 0
    for fold, data in daily.groupby("fold", sort=True):
        values = data[["price", "news", "zero", "N"]].to_numpy(dtype=float)
        n = len(values)
        starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block_size))))
        indices = ((starts[:, :, None] + np.arange(block_size)) % n).reshape(draws, -1)[:, :n]
        sampled = values[indices].sum(axis=1)
        rows.append(summarize(fold, values.sum(axis=0), sampled))
        pooled_point, pooled_draws = pooled_point + values.sum(axis=0), pooled_draws + sampled
    rows.append(summarize("Весь 2025", pooled_point, pooled_draws))
    return pd.DataFrame(rows)
