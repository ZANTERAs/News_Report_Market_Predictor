# News Market Bot — README (EN/ES)

> This README documents the Python script that fetches market news via RSS, performs sentiment analysis (VADER by default, optional FinBERT), maps articles to tickers, aggregates daily sentiment, and emits simple Buy/Hold/Sell signals with optional Plotly charts. fileciteturn0file0

---

## Overview (EN)

**News Market Bot** automates a lightweight news → sentiment → signal pipeline:
1. **Fetch** headlines from general finance RSS feeds and Yahoo Finance per‑ticker feeds.
2. **Clean & deduplicate** entries.
3. **Score sentiment** using VADER (fast, default) or FinBERT (finance‑tuned, optional).
4. **Map articles to tickers** by detecting tickers or company aliases in titles/summaries.
5. **Aggregate daily sentiment** per ticker and generate **Buy/Hold/Sell** signals.
6. **(Optional)** Backfill prices with `yfinance` and plot **sentiment vs. next‑day returns** (Plotly).

### Key Features
- RSS ingestion from multiple sources (general + per‑ticker).
- Robust ticker matching via regex + company aliases.
- Plug‑and‑play sentiment backends: **VADER** (NLTK) or **FinBERT** (Transformers).
- CSV artifacts: raw news, mapped+scored articles, daily signals.
- Optional Plotly charts; optional price join & fwd returns via `yfinance`.

### Folder & Files
- `main.py` (your script)
- `news_bot_output/`
  - `raw_news_*.csv` — raw deduped headlines
  - `mapped_scored_*.csv` — per‑article sentiment & ticker mapping
  - `daily_signals_*.csv` — daily mean sentiment, article counts, signals (+ price & fwd return if enabled)

---

## Setup (EN)

### 1) Python environment
```bash
python -m venv .venv
# Windows: .venv\Scriptsctivate
# macOS/Linux:
source .venv/bin/activate
pip install -U pip
pip install feedparser pandas nltk yfinance plotly
# Optional (heavier): FinBERT support
pip install torch transformers --upgrade
```

> **Colab/Jupyter tip:** If you run **inside notebooks**, either:
> - Execute as a shell: `!python main.py --tickers EXC XEL MSFT --days 5 --plot`, or
> - Modify `parse_args()` to use `parse_known_args()` (already included) so Jupyter’s `-f` flag is ignored.

### 2) NLTK/VADER
The script auto‑downloads the VADER lexicon on first run if missing.

---

## Usage (EN)

### Basic
```bash
python main.py --tickers EXC XEL AEP CEG MSFT GOOG --days 7
```

### Use FinBERT backend
```bash
python main.py --backend finbert --tickers EXC MSFT NVDA --days 5
```

### Plot sentiment vs next‑day returns
```bash
python main.py --plot --tickers EXC XEL AEP
```

### CLI Arguments
- `--tickers T1 T2 ...`  List of tickers (default includes EXC, XEL, AEP, CEG, MSFT, GOOG, AAPL, AMZN, NVDA).
- `--backend {vader,finbert}`  Sentiment backend (default: `vader`).
- `--days N`  News lookback window in days (default: 7).
- `--plot`  If set, shows Plotly charts.
- `--lookahead N`  Days ahead for forward returns (default: 1).

Outputs are written under `news_bot_output/` with timestamped filenames.

---

## Customization (EN)

### Add or edit ticker aliases
Improve matching by extending `TICKER_ALIASES` in the script, e.g. add brand names or products.

### Adjust signal thresholds
Change the simple rule inside `aggregate_daily()`:
- `>= +0.15` → BUY
- `<= -0.15` → SELL
- otherwise HOLD

### Add/Remove feeds
Append to `GENERAL_FEEDS` or per‑ticker sources. You can add Spanish or local feeds as needed.

---

## Troubleshooting (EN)

- **`SyntaxError: invalid syntax` when running `python -m venv`:** You ran a shell command _inside_ Python. Exit Python and run it in your terminal, or prefix with `!` in notebooks.
- **`unrecognized arguments: -f ... kernel-xxx.json` in notebooks:** Jupyter injects `-f`. The script uses `parse_known_args()` to ignore it; otherwise, run as `!python main.py ...`.
- **Missing packages:** `pip install feedparser pandas nltk yfinance plotly` (and `torch transformers` for FinBERT).
- **SSL/feed errors:** Some feeds may be rate‑limited or blocked; try again later or remove problematic feeds.

---

## Roadmap (EN)
- Weighted sentiment by source reliability and article freshness.
- Minimum article count filter per day (e.g., ignore `n_articles < 3`).
- Export a clean daily feature table (date, ticker, mean_sentiment, n_articles) for ML pipelines.
- Proper backtesting framework with time‑series splits.

---

# Descripción General (ES)

**News Market Bot** automatiza un flujo de **noticias → sentimiento → señal**:
1. **Descarga** titulares desde RSS financieros generales y por ticker (Yahoo Finance).
2. **Limpia y deduplica** entradas.
3. **Calcula sentimiento** con VADER (rápido, por defecto) o FinBERT (ajustado a finanzas, opcional).
4. **Asocia artículos a tickers** detectando tickers o alias de empresas en títulos/resúmenes.
5. **Agrega el sentimiento diario** por ticker y genera señales **Buy/Hold/Sell**.
6. **(Opcional)** Une precios con `yfinance` y grafica **sentimiento vs. retornos del día siguiente** (Plotly).

### Características
- Ingesta RSS multi‑fuente (general + por ticker).
- Matching robusto por regex + alias de compañía.
- Backends de sentimiento intercambiables: **VADER** o **FinBERT**.
- Artefactos CSV: noticias crudas, artículos mapeados y puntuados, señales diarias.
- Gráficos opcionales con Plotly y retornos futuros opcionales con `yfinance`.

---

## Instalación (ES)

```bash
python -m venv .venv
# Windows: .venv\Scriptsctivate
# macOS/Linux:
source .venv/bin/activate
pip install -U pip
pip install feedparser pandas nltk yfinance plotly
# Opcional (FinBERT):
pip install torch transformers --upgrade
```

> **Colab/Jupyter:** Ejecuta como `!python main.py ...` o usa `parse_known_args()` para ignorar el flag `-f` de Jupyter (ya incluido).

---

## Uso (ES)

### Básico
```bash
python main.py --tickers EXC XEL AEP CEG MSFT GOOG --days 7
```

### Con FinBERT
```bash
python main.py --backend finbert --tickers EXC MSFT NVDA --days 5
```

### Con gráficos
```bash
python main.py --plot --tickers EXC XEL AEP
```

### Argumentos
- `--tickers`  Lista de tickers.
- `--backend`  `vader` o `finbert`.
- `--days`  Ventana de días a considerar.
- `--plot`  Mostrar gráficos.
- `--lookahead`  Días para calcular el retorno futuro.

Los resultados se guardan en `news_bot_output/` con marca de tiempo.

---

## Personalización (ES)

- **Alias de tickers:** amplía `TICKER_ALIASES` para mejorar el matching (marcas, productos).
- **Umbrales de señal:** ajusta los cortes en `aggregate_daily()` (ej. `±0.10` o `±0.20`).
- **Feeds:** agrega medios locales o en español.

---

## Solución de problemas (ES)
- **Comandos de terminal dentro de Python:** ejecuta en la consola o usa `!` en notebooks.
- **Flags de Jupyter no reconocidos:** usa `!python main.py ...` o `parse_known_args()`.
- **Faltan paquetes:** instala dependencias con `pip`.

---

## License
MIT (suggested). Update as needed.

