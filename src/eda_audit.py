"""Read-only, chunked diagnostics for the Digikala exploration notebook."""
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import pandas as pd

MISSING = '<MISSING>'
MISSING_LABEL_VALUES = {'nan'}
CLASSES = ['recommended', 'not_recommended', 'no_idea']
BANDS = ['low', 'medium', 'high']
REQUIRED = ['id', 'body', 'rate', 'recommendation_status', 'product_id']
SIGNALS = {
    'blank_or_whitespace': r'^\s*$',
    'arabic_yeh_or_kaf': r'[يك]',
    'zwnj': '\u200c',
    'direction_marks': r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]',
    'repeated_whitespace': r'\s{2,}',
    'newline': r'[\r\n]',
    'url': r'(?i)(?:https?://|www\.)\S+',
    'html_like': r'<[^>]+>',
    'latin_letter': r'[A-Za-z]',
    'digit': r'[0-9۰-۹٠-٩]',
    'emoji_range': r'[\U0001F300-\U0001FAFF\u2600-\u27BF]',
    'repeated_character_3plus': r'(.)\1{2,}',
    'negation_candidate': r'(?:\bنه\b|\bنیست\b|\bندارد\b|نمی|نمي)',
}


def enrich(frame):
    """Diagnostic columns only: preserve original body and raw rate."""
    d = frame.copy()
    body = d['body'].fillna('')
    d['body_missing'] = d['body'].isna()
    d['characters'] = body.str.len().astype('int64')
    d['words'] = body.str.count(r'\S+').astype('int64')
    raw_label = d['recommendation_status'].astype('string')
    sentinel_missing = raw_label.str.strip().str.lower().isin(MISSING_LABEL_VALUES)
    d['label'] = raw_label.mask(sentinel_missing).fillna(MISSING)
    d['rate_numeric'] = pd.to_numeric(d['rate'], errors='coerce')
    valid = d['rate_numeric'].between(0, 5)
    d['rating_band'] = pd.Series('invalid', index=d.index, dtype='string')
    d.loc[valid & d.rate_numeric.lt(2.5), 'rating_band'] = 'low'
    d.loc[valid & d.rate_numeric.ge(2.5) & d.rate_numeric.lt(4), 'rating_band'] = 'medium'
    d.loc[valid & d.rate_numeric.ge(4), 'rating_band'] = 'high'
    for name, pattern in SIGNALS.items():
        # findall/count avoids pandas' capturing-group warning for repeated chars.
        d[name] = body.str.count(pattern).gt(0)
    d['blank_or_whitespace'] &= ~d['body_missing']
    d['very_short_1_2_words'] = d.words.between(1, 2)
    d['long_over_500_words'] = d.words.gt(500)
    return d


def frequency_table(counts):
    s = pd.Series(dict(counts), dtype='int64').sort_values(ascending=False)
    return pd.DataFrame({'count': s, 'percent': 100 * s / max(s.sum(), 1)})


def histogram_summary(counts):
    """Exact pandas-style linear quantiles from an integer frequency table."""
    if not counts:
        return pd.Series(dtype='float64')
    x = np.array(sorted(counts), dtype=float)
    w = np.array([counts[k] for k in sorted(counts)], dtype=np.int64)
    n = int(w.sum())
    cumulative = w.cumsum()
    mean = float(np.dot(x, w) / n)
    result = {'count': n, 'min': x[0], 'mean': mean,
              'std': np.sqrt(np.dot((x - mean) ** 2, w) / (n - 1)) if n > 1 else np.nan,
              'max': x[-1]}
    for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]:
        pos = (n - 1) * q
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        a, b = x[np.searchsorted(cumulative, [lo + 1, hi + 1])]
        result[f'p{q * 100:g}'] = a + (b - a) * (pos - lo)
    return pd.Series(result)


def scan_csv(path, chunksize=100_000, sample_size=20_000, seed=2026, progress=True):
    """Full-file aggregations; bounded uniform sample via seeded random priorities.

    CSV NA policy is explicit: only empty fields are missing. Literal 'NA',
    'null', etc. remain visible for inspection. Parser errors are never skipped.
    """
    start = time.monotonic()
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    absent = set(REQUIRED) - set(columns)
    if absent:
        raise ValueError(f'Missing required columns: {sorted(absent)}')
    counts = {k: Counter() for k in ['missing', 'blank', 'labels', 'raw_labels', 'rates',
                                   'bands', 'products', 'sellers', 'signals', 'quality']}
    cross = defaultdict(Counter)
    lengths = {'characters': Counter(), 'words': Counter()}
    by_label = defaultdict(lambda: {'characters': Counter(), 'words': Counter(), 'signals': Counter()})
    rng = np.random.default_rng(seed)
    sample = None
    examples = defaultdict(list)
    n = 0
    for chunk in pd.read_csv(path, dtype='string', keep_default_na=False,
                             na_values=[''], chunksize=chunksize, on_bad_lines='error'):
        n += len(chunk)
        counts['missing'].update(chunk.isna().sum().to_dict())
        # The source file uses the literal text "nan" for missing recommendation labels.
        label_sentinel_missing = chunk['recommendation_status'].astype('string').str.strip().str.lower().isin(MISSING_LABEL_VALUES)
        counts['missing']['recommendation_status'] += int(label_sentinel_missing.sum())
        counts['blank'].update({c: int(chunk[c].str.fullmatch(r'\s*').fillna(False).sum()) for c in columns})
        d = enrich(chunk)
        counts['labels'].update(d.label.value_counts().to_dict())
        counts['raw_labels'].update(d.recommendation_status.dropna().value_counts().to_dict())
        r = d.rate_numeric
        finite = r.notna() & np.isfinite(r.fillna(np.nan).astype(float))
        valid = finite & r.between(0, 5)
        counts['rates'].update(r[finite].value_counts().to_dict())
        counts['bands'].update(d.rating_band.value_counts().to_dict())
        counts['products'].update(d.product_id.dropna().value_counts().to_dict())
        if 'seller_code' in d:
            counts['sellers'].update(d.seller_code.dropna().value_counts().to_dict())
        quality = {
            'rate_missing': d.rate.isna(),
            'rate_non_numeric': d.rate.notna() & r.isna(),
            'rate_nonfinite': r.notna() & ~finite,
            'rate_out_of_range': finite & ~r.between(0, 5),
            'rate_zero': r.eq(0),
            'rate_fractional_valid': valid & r.mod(1).ne(0),
            'body_missing': d.body_missing,
            'unknown_label': ~d.label.isin(CLASSES + [MISSING]),
            'missing_product': d.product_id.isna(),
            'missing_id': d.id.isna(),
        }
        counts['quality'].update({k: int(v.sum()) for k, v in quality.items()})
        flags = list(SIGNALS) + ['very_short_1_2_words', 'long_over_500_words']
        counts['signals'].update(d[flags].sum().to_dict())
        for metric in lengths:
            lengths[metric].update(d.loc[~d.body_missing, metric].value_counts().to_dict())
        for label, g in d.groupby('label', sort=False):
            for metric in lengths:
                by_label[label][metric].update(g.loc[~g.body_missing, metric].value_counts().to_dict())
            by_label[label]['signals'].update(g[flags].sum().to_dict())
        rate_bucket = r.astype('string').fillna('<INVALID_OR_MISSING>')
        cross['rate_label'].update(pd.DataFrame({'rate': rate_bucket, 'label': d.label}).value_counts().to_dict())
        cross['band_label'].update(d[['rating_band', 'label']].value_counts().to_dict())
        for col in ['is_buyer']:
            if col in d:
                cross[col + '_label'].update(pd.DataFrame({col: d[col].fillna(MISSING), 'label': d.label}).value_counts().to_dict())
        for name, mask in {**quality, **{k: d[k] for k in flags}}.items():
            if len(examples[name]) < 12:
                examples[name].extend(d.loc[mask.fillna(False), REQUIRED].head(12 - len(examples[name])).to_dict('records'))
        # Repeated bottom-k selection equals bottom-k over every row seen.
        d['_priority'] = rng.random(len(d))
        candidate = d.nsmallest(min(sample_size, len(d)), '_priority')
        sample = candidate if sample is None else pd.concat([sample, candidate], ignore_index=True).nsmallest(sample_size, '_priority')
        if progress and (n == len(chunk) or n % (chunksize * 10) == 0):
            print(f'{n:,} rows scanned | {time.monotonic() - start:.1f}s', flush=True)
    return {'rows': n, 'columns': columns, 'counts': counts, 'cross': dict(cross),
            'lengths': lengths, 'by_label': dict(by_label),
            'sample': sample.drop(columns='_priority').reset_index(drop=True) if sample is not None else pd.DataFrame(),
            'examples': dict(examples), 'seconds': time.monotonic() - start,
            'seed': seed, 'sample_size': sample_size,
            'na_policy': 'Only empty CSV fields are missing; whitespace is audited separately.'}


def cross_table(counts):
    if not counts:
        return pd.DataFrame()
    s = pd.Series(dict(counts))
    s.index = pd.MultiIndex.from_tuples(s.index)
    return s.unstack(fill_value=0)


def normalization_candidate(s):
    """Preview only. Preserve ZWNJ, emoji, punctuation, numbers and negation."""
    s = str(s).translate(str.maketrans({'ي': 'ی', 'ك': 'ک'}))
    s = re.sub(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def duplicate_audit(path, db_path, chunksize=100_000, progress=True):
    """Exact on-disk grouping, no body hashes and no changes to source data.

    Can require several GB of disk and minutes to complete. Blank bodies are
    excluded from body duplication; missing IDs are excluded from ID grouping.
    Conflicts mean distinct valid targets on an identical raw body.
    """
    db_path = Path(db_path)
    started = time.monotonic()
    if db_path.exists():
        # Reuse a completed audit database so rerunning the notebook is safe.
        with closing(sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro', uri=True)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            required = {'reviews', 'body_groups', 'id_groups'}
            if not required.issubset(tables):
                raise ValueError(f'?????? ????? ???? ???? ???????? ???? ???? ?????: {sorted(required - tables)}')
            n = conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
            if n == 0:
                raise ValueError(f'?????? ????? ???? ???: {db_path}')
            summary = pd.read_sql_query('''SELECT
                (SELECT COUNT(*) FROM id_groups WHERE n>1) AS duplicate_id_groups,
                (SELECT COALESCE(SUM(n-1),0) FROM id_groups WHERE n>1) AS excess_id_rows,
                (SELECT COUNT(*) FROM body_groups WHERE n>1) AS duplicate_body_groups,
                (SELECT COALESCE(SUM(n-1),0) FROM body_groups WHERE n>1) AS excess_body_rows,
                (SELECT COUNT(*) FROM body_groups WHERE labels>1) AS recommendation_conflict_groups,
                (SELECT COUNT(*) FROM body_groups WHERE bands>1) AS rating_band_conflict_groups''', conn)
            repeated = pd.read_sql_query('SELECT * FROM body_groups WHERE n>1 ORDER BY n DESC, body LIMIT 30', conn)
            conflicts = pd.read_sql_query('SELECT * FROM body_groups WHERE labels>1 OR bands>1 ORDER BY n DESC, body LIMIT 30', conn)
        return {'summary': summary, 'repeated': repeated, 'conflicts': conflicts,
                'rows': n, 'seconds': time.monotonic() - started,
                'database': str(db_path), 'reused_existing': True}
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute('PRAGMA temp_store=FILE')
        conn.execute('CREATE TABLE reviews (id TEXT, body TEXT, label TEXT, band TEXT)')
        n = 0
        for c in pd.read_csv(path, usecols=REQUIRED, dtype='string', keep_default_na=False,
                             na_values=[''], chunksize=chunksize, on_bad_lines='error'):
            r = pd.to_numeric(c.rate, errors='coerce')
            bands = pd.Series(pd.NA, index=c.index, dtype='string')
            bands.loc[r.between(0, 5) & r.lt(2.5)] = 'low'
            bands.loc[r.ge(2.5) & r.lt(4)] = 'medium'
            bands.loc[r.ge(4) & r.le(5)] = 'high'
            records = pd.DataFrame({'id': c.id,
                                    'body': c.body.mask(c.body.fillna('').str.strip().eq('')),
                                    'label': c.recommendation_status.where(c.recommendation_status.isin(CLASSES)),
                                    'band': bands})
            records = records.astype(object).where(records.notna(), None)
            conn.executemany('INSERT INTO reviews VALUES (?,?,?,?)', records.itertuples(index=False, name=None))
            conn.commit()
            n += len(c)
            if progress and n % (chunksize * 10) == 0:
                print(f'{n:,} rows written for exact duplicate audit', flush=True)
        conn.execute('CREATE TABLE body_groups AS SELECT body, COUNT(*) AS n, COUNT(DISTINCT label) AS labels, COUNT(DISTINCT band) AS bands FROM reviews WHERE body IS NOT NULL GROUP BY body')
        conn.execute('CREATE TABLE id_groups AS SELECT id, COUNT(*) AS n FROM reviews WHERE id IS NOT NULL GROUP BY id')
        summary = pd.read_sql_query('''SELECT
            (SELECT COUNT(*) FROM id_groups WHERE n>1) AS duplicate_id_groups,
            (SELECT COALESCE(SUM(n-1),0) FROM id_groups WHERE n>1) AS excess_id_rows,
            (SELECT COUNT(*) FROM body_groups WHERE n>1) AS duplicate_body_groups,
            (SELECT COALESCE(SUM(n-1),0) FROM body_groups WHERE n>1) AS excess_body_rows,
            (SELECT COUNT(*) FROM body_groups WHERE labels>1) AS recommendation_conflict_groups,
            (SELECT COUNT(*) FROM body_groups WHERE bands>1) AS rating_band_conflict_groups''', conn)
        repeated = pd.read_sql_query('SELECT * FROM body_groups WHERE n>1 ORDER BY n DESC, body LIMIT 30', conn)
        conflicts = pd.read_sql_query('SELECT * FROM body_groups WHERE labels>1 OR bands>1 ORDER BY n DESC, body LIMIT 30', conn)
    return {'summary': summary, 'repeated': repeated, 'conflicts': conflicts,
            'rows': n, 'seconds': time.monotonic() - started, 'database': str(db_path)}
