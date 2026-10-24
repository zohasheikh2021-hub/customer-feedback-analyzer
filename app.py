"""Customer Feedback Analyzer: Streamlit front end.

The app only *loads* the models saved by ``train_models.py`` and renders their predictions.
It never trains anything. All numbers shown in the analytics sections are read from
``reports/results.json`` and ``reports/eda_summary.json`` (written by the training run).
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

import config as cfg
from predictor import (FeedbackAnalysis, FeedbackPredictor, InferenceError, InvalidInputError,
                       ModelLoadError, ModelNotFoundError, TaskPrediction)

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon=str(cfg.ASSETS_DIR / "favicon.png"),
                   layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- logging
def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("feedback_analyzer")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        try:
            cfg.LOG_DIR.mkdir(exist_ok=True)
            fh = logging.FileHandler(cfg.LOG_DIR / "app.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass  # read-only deployments: console logging is enough
    return logger


log = _setup_logging()

# --------------------------------------------------------------------------- constants
EXAMPLES = [
    ("Late and damaged", "The product arrived two days late, the package was damaged, and customer support has not responded."),
    ("Double charge", "The checkout page charged me twice for one order and I still have not received a refund."),
    ("Happy customer", "I am very happy with the new app. Setup was easy and the support team answered my question within minutes."),
    ("Order question", "I would like to know whether my order can still be changed. The website shows that it has not shipped yet."),
]

ICONS = {
    "logo": '<path d="M21 15a2 2 0 0 1-2 2H8l-5 4V6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M8 10h8M8 13h5"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "smile": '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><path d="M9 9h.01M15 9h.01"/>',
    "tag": '<path d="M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0L2 12V2h10l8.6 8.6a2 2 0 0 1 0 2.8z"/><path d="M7 7h.01"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><path d="M4 22v-7"/>',
    "chart": '<path d="M12 20V10M18 20V4M6 20v-4"/>',
    "layers": '<path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    "filter": '<path d="M22 3H2l8 9.5V19l4 2v-8.5z"/>',
    "type": '<path d="M4 7V4h16v3M9 20h6M12 4v16"/>',
    "network": '<circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/><path d="M7 6l3.2 4.8M7 18l3.2-4.8M14 12h3"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "check": '<path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/><path d="m22 4-10 10-3-3"/>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "split": '<path d="M16 3h5v5M8 3H3v5M21 3l-7.5 7.5M3 3l7.5 7.5M12 12v9"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
}


def icon(name: str) -> str:
    return f'<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">{ICONS[name]}</svg>'


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def pct(p: float) -> str:
    if p < 0.005:
        return "<1%"
    return f"{p * 100:.0f}%"


def short_name(algo: str) -> str:
    return algo.replace(" (tuned)", "")


def md(markup: str) -> None:
    """Render trusted, app-generated HTML. User text must always pass through ``esc`` first."""
    st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------------- cached loading
@st.cache_resource(show_spinner=False)
def _load_predictor() -> FeedbackPredictor:
    return FeedbackPredictor()


def get_predictor():
    """Return (predictor, error_message). Failures are not cached, so fixing the files and rerunning works."""
    try:
        return _load_predictor(), None
    except (ModelNotFoundError, ModelLoadError) as exc:
        log.error("Model loading problem: %s", exc)
        return None, str(exc)
    except Exception:  # noqa: BLE001
        log.exception("Unexpected error while loading models")
        return None, "The application could not start the models. Check the server log for details."


@st.cache_data(show_spinner=False)
def _read_json(path_str: str, mtime: float) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def read_report(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return _read_json(str(path), path.stat().st_mtime)
    except (OSError, json.JSONDecodeError):
        log.exception("Could not read %s", path.name)
        return None


@st.cache_data(show_spinner=False)
def _read_css(mtime: float) -> str:
    return cfg.CSS_PATH.read_text(encoding="utf-8")


def inject_css() -> None:
    try:
        st.markdown(f"<style>{_read_css(cfg.CSS_PATH.stat().st_mtime)}</style>", unsafe_allow_html=True)
    except OSError:
        log.warning("Stylesheet missing; the app continues with the default theme.")


# --------------------------------------------------------------------------- state callbacks
def _init_state() -> None:
    st.session_state.setdefault("feedback_text", "")
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("error", None)


def use_example(index: int) -> None:
    st.session_state.feedback_text = EXAMPLES[index][1]
    st.session_state.analysis = None
    st.session_state.error = None


def clear_all() -> None:
    st.session_state.feedback_text = ""
    st.session_state.analysis = None
    st.session_state.error = None


# --------------------------------------------------------------------------- html components
def notice(kind: str, text: str) -> None:
    ico = {"error": "alert", "warn": "alert", "success": "check", "info": "info"}[kind]
    role = "alert" if kind in ("error", "warn") else "status"
    md(f'<div class="notice {kind}" role="{role}">{icon(ico)}<div>{esc(text)}</div></div>')


def render_header(models_ok: bool) -> None:
    pill = ('<span class="status-pill" title="All prediction models were loaded and checked."><span class="dot"></span>Models loaded</span>'
            if models_ok else
            '<span class="status-pill bad" title="Trained model files are missing or unreadable."><span class="dot"></span>Models unavailable</span>')
    md(f'''<header class="app-header"><div class="brand"><div class="brand-mark">{icon("logo")}</div>
<div><div class="brand-name">Customer Feedback Analyzer</div><div class="brand-tag">Sentiment, complaint type and priority from one message</div></div></div>
<div class="header-right"><nav class="nav" aria-label="Page sections"><a href="#analyzer">Analyzer</a><a href="#analytics">Analytics</a><a href="#models">How it works</a></nav>{pill}</div></header>''')


def render_hero(predictor: Optional[FeedbackPredictor], eda: Optional[dict]) -> None:
    if predictor is not None:
        m = predictor.metadata["models"]
        sent_m, comp_m = short_name(m["sentiment"]["algorithm"]), short_name(m["complaint_type"]["algorithm"])
        prio_m = "Keras neural network"
    else:
        sent_m = comp_m = "Machine learning model"
        prio_m = "Keras neural network"
    facts = ""
    if eda:
        facts = (f'<div class="hero-facts"><span><b>{eda["n_rows"]:,}</b> historical records</span>'
                 f'<span><b>{len(eda["distributions"]["complaint_type"])}</b> complaint categories</span>'
                 f'<span><b>3</b> predictions per message</span></div>')
    md(f'''<section class="hero"><div><h1>Understand Every Customer Voice</h1>
<p class="lead">Transform customer feedback into actionable insights using NLP, machine learning, and neural networks.</p>
<a class="cta" href="#analyzer">Analyze feedback</a>{facts}</div>
<div class="card flow" aria-label="How one message flows through the models"><p class="flow-title">What happens to one message</p>
<p class="flow-sub">The same text goes through three trained models.</p>
<div class="flow-input">{icon("message")}<span>Customer feedback text</span></div>
<div class="flow-step">Cleaned, then turned into TF-IDF numbers</div>
<div class="flow-out">
<div class="flow-row" style="--rail:var(--positive);border-left-color:var(--positive)">{icon("smile")}<div><b>Sentiment</b><span>{esc(sent_m)}</span></div></div>
<div class="flow-row" style="--rail:var(--accent);border-left-color:var(--accent)">{icon("tag")}<div><b>Complaint type</b><span>{esc(comp_m)}</span></div></div>
<div class="flow-row" style="--rail:var(--medium);border-left-color:var(--medium)">{icon("flag")}<div><b>Priority</b><span>{esc(prio_m)}</span></div></div>
</div></div></section>''')


def _probability_rows(pred: TaskPrediction, top: Optional[int]) -> str:
    items = sorted((pred.probabilities or {}).items(), key=lambda kv: -kv[1])
    if top:
        items = items[:top]
    rows = "".join(
        f'<div class="prob-row"><span class="name" title="{esc(k)}">{esc(k)}</span>'
        f'<span class="prob-bar"><i style="width:{max(v * 100, 1.5):.1f}%"></i></span><span class="pct">{pct(v)}</span></div>'
        for k, v in items)
    return f'<div class="prob-list">{rows}</div>'


def _ring(conf: Optional[float]) -> str:
    tip = ("Confidence is the model's estimated probability for this label. It is an estimate learned from "
           "historical data, not a guarantee that the label is correct.")
    if conf is None:
        return f'<div class="ring na" title="{esc(tip)}"><div class="ring-inner">Confidence unavailable</div></div>'
    return (f'<div class="ring" style="--p:{conf * 100:.1f}" title="{esc(tip)}" role="img" aria-label="Confidence {pct(conf)}">'
            f'<div class="ring-inner"><div><b>{pct(conf)}</b><small>confidence</small></div></div></div>')


def result_card(title: str, ico: str, pred: TaskPrediction, tone: str, foot: str, top: Optional[int] = 3,
                badge: str = "") -> str:
    second = ""
    if pred.probabilities and len(pred.probabilities) > 1 and not badge:
        ordered = sorted(pred.probabilities.items(), key=lambda kv: -kv[1])
        second = f'<div class="runner">Next most likely: <b>{esc(ordered[1][0])}</b> ({pct(ordered[1][1])})</div>'
    probs = _probability_rows(pred, top) if pred.probabilities else '<div class="runner">Class probabilities unavailable</div>'
    return (f'<article class="result-card" style="--tone:{tone}" aria-label="{esc(title)} result">'
            f'<div class="rc-head">{icon(ico)}<span>{esc(title)}</span></div>'
            f'<div class="rc-body"><div><div class="rc-label">{esc(pred.label)}</div>{badge}{second}</div>{_ring(pred.confidence)}</div>'
            f'{probs}<div class="rc-foot">{foot}</div></article>')


def render_results(a: FeedbackAnalysis, meta: dict) -> None:
    sent_tone = cfg.SENTIMENT_COLORS.get(a.sentiment.label, cfg.COLORS["accent"])
    prio_tone = cfg.PRIORITY_COLORS.get(a.priority.label, cfg.COLORS["accent"])
    level = {"Low": 1, "Medium": 2, "High": 3}.get(a.priority.label, 0)
    meter = "".join(f'<i class="{"on" if i < level else ""}"></i>' for i in range(3))
    badge = f'<div class="badge"><span class="meter" aria-hidden="true">{meter}</span>{esc(a.priority.label)} priority</div>'
    m = meta["models"]
    snippet = a.text if len(a.text) <= 110 else a.text[:107] + "..."
    md(f'<div class="result-for">Results for: <b>{esc(snippet)}</b></div>'
       f'<div class="results-grid">'
       + result_card("Sentiment", "smile", a.sentiment, sent_tone, f"{esc(short_name(m['sentiment']['algorithm']))} with TF-IDF", top=3)
       + result_card("Complaint type", "tag", a.complaint_type, cfg.COLORS["accent"], f"{esc(short_name(m['complaint_type']['algorithm']))} with TF-IDF", top=3)
       + result_card("Priority", "flag", a.priority, prio_tone, "Keras feed-forward neural network", top=3, badge=badge)
       + '</div>')
    md('<p class="disclaimer">These predictions are estimates learned from historical data. A confidence score shows how strongly '
       'a model favors a label; it does not guarantee that the label is correct.</p>')


# --------------------------------------------------------------------------- plotly helpers
FONT_BODY = "Inter, Segoe UI, system-ui, sans-serif"
FONT_DISPLAY = "Sora, Segoe UI, system-ui, sans-serif"
GRID = "rgba(148,163,184,0.14)"


def style_fig(fig: go.Figure, title: str, height: int = 330, legend: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(family=FONT_DISPLAY, size=15, color="#e8ebf7")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
        font=dict(family=FONT_BODY, color="#c5cbe8", size=12), margin=dict(l=8, r=12, t=52, b=8),
        hoverlabel=dict(bgcolor="#19224a", bordercolor="#6366f1", font=dict(family=FONT_BODY, size=12, color="#e8ebf7")),
        showlegend=legend, legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=11))
    return fig


def show(fig: go.Figure, key: str) -> None:
    with st.container(key=f"chart_{key}"):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True}, key=f"plot_{key}")


def probability_figure(pred: TaskPrediction, title: str, colors: Dict[str, str], key: str) -> None:
    items = sorted((pred.probabilities or {}).items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        marker_color=[colors.get(k, cfg.COLORS["accent"]) for k, _ in items],
        text=[pct(v) for _, v in items], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
    show(style_fig(fig, title, height=max(200, 62 + 34 * len(items))), key)


def render_details(a: FeedbackAnalysis, predictor: FeedbackPredictor, meta: dict) -> None:
    with st.expander("Analysis details", expanded=False):
        base = a.baseline_priority
        m = meta["models"]
        tm = meta["test_metrics"]
        md(f'''<dl class="kv">
<dt>Submitted feedback</dt><dd><div class="quote">{esc(a.text)}</div></dd>
<dt>Text the models see</dt><dd><div class="quote">{esc(a.cleaned_text)}</div></dd>
<dt>Vocabulary match</dt><dd>{a.vocabulary_coverage:.0%} of the meaningful words appear in the training vocabulary</dd>
<dt>Sentiment model</dt><dd>{esc(m["sentiment"]["algorithm"])}; test accuracy {tm["sentiment"]["accuracy"]:.1%} on historical data</dd>
<dt>Complaint model</dt><dd>{esc(m["complaint_type"]["algorithm"])}; test accuracy {tm["complaint_type"]["accuracy"]:.1%} on historical data</dd>
<dt>Priority model</dt><dd>Keras ANN; test accuracy {tm["priority_ann"]["accuracy"]:.1%} on historical data</dd>
<dt>Priority reference</dt><dd>{esc(short_name(m["priority"]["baseline_algorithm"]))} predicts <b>{esc(base.label)}</b> ({pct(base.confidence or 0)}); test accuracy {tm["priority_baseline"]["accuracy"]:.1%}</dd>
</dl>''')
        for w in a.warnings:
            notice("warn", w)
        c1, c2, c3 = st.columns(3)
        with c1:
            probability_figure(a.sentiment, "Sentiment probabilities", cfg.SENTIMENT_COLORS, "p_sent")
        with c2:
            probability_figure(a.priority, "Priority probabilities (ANN)", cfg.PRIORITY_COLORS, "p_prio")
        with c3:
            probability_figure(a.complaint_type, "Complaint type probabilities", {}, "p_comp")
        md('<p class="disclaimer">Test accuracies describe performance on held-out historical records, not on this message.</p>')


# --------------------------------------------------------------------------- analyzer section
def render_analyzer(predictor: Optional[FeedbackPredictor], meta: Optional[dict], load_error: Optional[str]) -> None:
    ss = st.session_state
    md('<span class="anchor" id="analyzer"></span><h2 class="section-title" style="margin-top:1.4rem">Feedback analyzer</h2>'
       '<p class="section-sub">Paste a review or complaint, or start from an example. The three models run when you press Analyze feedback.</p>')
    if load_error:
        notice("error", load_error)

    with st.container(border=False):
        st.text_area("Customer feedback", key="feedback_text", height=170, max_chars=cfg.MAX_INPUT_CHARS,
                     placeholder="Type or paste a customer review or complaint, for example: The delivery was late and nobody answered my emails...",
                     help=f"Up to {cfg.MAX_INPUT_CHARS:,} characters. Only the text is analyzed; nothing is stored.")
        md('<div class="examples-label">Try an example</div>')
        cols = st.columns(len(EXAMPLES))
        for i, (label, _) in enumerate(EXAMPLES):
            cols[i].button(label, key=f"ex_{i}", on_click=use_example, args=(i,), width="stretch",
                           help="Fills the text box with this example. Press Analyze feedback to run the models.")
        b1, b2, _sp = st.columns([1.1, 0.8, 2.6])
        analyze = b1.button("Analyze feedback", key="analyze_btn", width="stretch",
                            help="Runs the sentiment, complaint type and priority models on your text.")
        b2.button("Clear", key="clear_btn", on_click=clear_all, width="stretch", help="Empties the text box and removes the results.")

    if analyze:
        if predictor is None:
            ss.error = load_error or "The models are not available."
            ss.analysis = None
        else:
            try:
                with st.spinner("Running the three models..."):
                    ss.analysis = predictor.analyze(ss.feedback_text)
                ss.error = None
            except InvalidInputError as exc:
                ss.analysis, ss.error = None, str(exc)
            except InferenceError as exc:
                log.error("Inference error: %s", exc)
                ss.analysis, ss.error = None, str(exc)
            except Exception:  # noqa: BLE001
                log.exception("Unexpected analysis failure")
                ss.analysis, ss.error = None, "Something unexpected went wrong. Please try again."

    if ss.error and not load_error:
        notice("error", ss.error)
    if ss.analysis is not None and predictor is not None and meta is not None:
        a: FeedbackAnalysis = ss.analysis
        notice("success", "Analysis complete. The three predictions are shown below.")
        for w in a.warnings:
            notice("warn", w)
        render_results(a, meta)
        render_details(a, predictor, meta)


# --------------------------------------------------------------------------- analytics section
def bar_distribution(counts: Dict[str, int], title: str, colors, order: Optional[List[str]] = None,
                     horizontal: bool = False, key: str = "") -> None:
    order = order or sorted(counts, key=counts.get, reverse=True)
    total = sum(counts.values())
    vals = [counts[k] for k in order]
    colour = [colors[k] for k in order] if isinstance(colors, dict) else colors[: len(order)]
    custom = [v / total for v in vals]
    if horizontal:
        fig = go.Figure(go.Bar(x=vals[::-1], y=order[::-1], orientation="h", marker_color=colour[::-1], customdata=custom[::-1],
                               text=[f"{v:,}" for v in vals[::-1]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:,} records (%{customdata:.1%})<extra></extra>"))
        fig.update_xaxes(range=[0, max(vals) * 1.18])
    else:
        fig = go.Figure(go.Bar(x=order, y=vals, marker_color=colour, customdata=custom,
                               text=[f"{v:,}" for v in vals], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}<br>%{y:,} records (%{customdata:.1%})<extra></extra>"))
        fig.update_yaxes(range=[0, max(vals) * 1.18])
    show(style_fig(fig, title, height=330 if not horizontal else 360), key)


def heatmap(z, x, y, text, title: str, key: str, height: int = 340, normalized_label: str = "Share of row") -> None:
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, text=text, texttemplate="%{text:,}", zmin=0, zmax=1,
        colorscale=[[0, "#121a36"], [0.5, "#4338ca"], [1, "#a78bfa"]], showscale=False, xgap=3, ygap=3,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>%{text:,} records (%{z:.0%} of row)<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style_fig(fig, title, height=height), key)


def metric_card(label: str, value: str, note: str) -> str:
    return f'<div class="metric"><div class="m-label">{esc(label)}</div><div class="m-value">{esc(value)}</div><div class="m-note">{esc(note)}</div></div>'


def table_html(headers: List[str], rows: List[List[str]], best_cols: Optional[Dict[int, float]] = None) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>"
    return f'<div class="table-wrap"><table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_analytics(results: Optional[dict], eda: Optional[dict]) -> None:
    md('<span class="anchor" id="analytics"></span><h2 class="section-title">Analytics</h2>'
       '<p class="section-sub">Everything here is measured from the 30,000 historical records and the held-out test set. '
       'It describes the training data and model quality, not your current message.</p>'
       f'<div class="scope-note">{icon("database")}<span>Historical dataset statistics and test results</span></div>')
    if not results or not eda:
        notice("warn", "Analytics are unavailable because the report files are missing. Run `python train_models.py` to create them.")
        return
    tasks = results["tasks"]
    s_sel = tasks["sentiment"]["candidates"][tasks["sentiment"]["selected"]]["test"]
    c_sel = tasks["complaint_type"]["candidates"][tasks["complaint_type"]["selected"]]["test"]
    ann = tasks["priority"]["ann"]["test"]
    base_name = tasks["priority"]["selected"]
    base = tasks["priority"]["candidates"][base_name]["test"]
    md('<div class="metric-grid">'
       + metric_card("Historical records", f"{eda['n_rows']:,}", f"{eda['duplicates']['distinct_texts']:,} distinct texts")
       + metric_card("Sentiment macro-F1", f"{s_sel['macro_f1']:.3f}", f"test set, accuracy {s_sel['accuracy']:.1%}")
       + metric_card("Complaint type macro-F1", f"{c_sel['macro_f1']:.3f}", f"test set, accuracy {c_sel['accuracy']:.1%}")
       + metric_card("Priority ANN accuracy", f"{ann['accuracy']:.1%}", f"test set, macro-F1 {ann['macro_f1']:.3f}")
       + '</div>')

    t_data, t_perf, t_cm = st.tabs(["Dataset", "Model performance", "Confusion matrices"])

    with t_data:
        d = eda["distributions"]
        c1, c2 = st.columns(2)
        with c1:
            bar_distribution(d["sentiment"], "Sentiment distribution", cfg.SENTIMENT_COLORS, cfg.SENTIMENT_ORDER, key="d_sent")
        with c2:
            bar_distribution(d["priority"], "Priority distribution", cfg.PRIORITY_COLORS, cfg.PRIORITY_ORDER, key="d_prio")
        c3, c4 = st.columns(2)
        with c3:
            bar_distribution(d["complaint_type"], "Complaint type distribution", cfg.CATEGORY_PALETTE, horizontal=True, key="d_comp")
        with c4:
            words = eda["top_words"][:12]
            fig = go.Figure(go.Bar(x=[c for _, c in words][::-1], y=[w for w, _ in words][::-1], orientation="h",
                                   marker_color=cfg.COLORS["accent_2"], hovertemplate="%{y}: %{x:,} occurrences<extra></extra>"))
            show(style_fig(fig, "Most frequent words after cleaning", height=360), "d_words")
        ct = eda["crosstab_sentiment_priority"]
        z = [[ct[s][p] / sum(ct[s].values()) for p in cfg.PRIORITY_ORDER] for s in cfg.SENTIMENT_ORDER]
        txt = [[ct[s][p] for p in cfg.PRIORITY_ORDER] for s in cfg.SENTIMENT_ORDER]
        fig = go.Figure(go.Heatmap(z=z, x=cfg.PRIORITY_ORDER, y=cfg.SENTIMENT_ORDER, text=txt, texttemplate="%{text:,}", zmin=0, zmax=0.7,
                                   colorscale=[[0, "#121a36"], [0.5, "#4338ca"], [1, "#a78bfa"]], showscale=False, xgap=3, ygap=3,
                                   hovertemplate="%{y} / %{x}<br>%{text:,} records (%{z:.0%} of the sentiment group)<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        c5, c6 = st.columns(2)
        with c5:
            show(style_fig(fig, "Priority by sentiment (colour = share of the row)", height=330), "d_xtab")
        with c6:
            lab = eda["label_conflicts_per_text"]
            md('<div class="card info-card"><h4>' + icon("info") + 'What the data tell us</h4>'
               + "".join(f"<p>{esc(o)}</p>" for o in eda["observations"][1:4] + eda["observations"][4:6]) + '</div>')

    with t_perf:
        rows = []
        for task, key in [("Sentiment", "sentiment"), ("Complaint type", "complaint_type"), ("Priority", "priority")]:
            for name, r in tasks[key]["candidates"].items():
                rows.append((task, name, r["test"]["macro_f1"], r["test"]["accuracy"], r["cv"]["f1_macro"]["mean"]))
        rows.append(("Priority", "Keras ANN", ann["macro_f1"], ann["accuracy"], None))
        models = ["Logistic Regression (tuned)", "Multinomial Naive Bayes", "Linear SVM", "Majority-class baseline", "Keras ANN"]
        palette = ["#6366f1", "#8b5cf6", "#06b6d4", "#64748b", "#f59e0b"]
        fig = go.Figure()
        for mname, col in zip(models, palette):
            sel = [r for r in rows if r[1] == mname]
            fig.add_bar(name=mname, x=[r[0] for r in sel], y=[r[2] for r in sel], marker_color=col,
                        text=[f"{r[2]:.2f}" for r in sel], textposition="outside", cliponaxis=False,
                        customdata=[[r[3], r[4] if r[4] is not None else float("nan")] for r in sel],
                        hovertemplate="%{x}: %{fullData.name}<br>Test macro-F1 %{y:.3f}<br>Test accuracy %{customdata[0]:.3f}<extra></extra>")
        fig.update_layout(barmode="group", bargap=0.22)
        fig.update_yaxes(range=[0, 1.12], title="Test macro-F1")
        show(style_fig(fig, "Model comparison on the held-out test set", height=400, legend=True), "m_compare")

        def row(label, ev, cv=None):
            return [label, f"{ev['accuracy']:.3f}", f"{ev['macro_precision']:.3f}", f"{ev['macro_recall']:.3f}", f"{ev['macro_f1']:.3f}",
                    f"{cv['mean']:.3f} ± {cv['std']:.3f}" if cv else "not applicable"]
        t_rows = [
            row(f"Sentiment: {short_name(tasks['sentiment']['selected'])}", s_sel, tasks["sentiment"]["candidates"][tasks["sentiment"]["selected"]]["cv"]["f1_macro"]),
            row(f"Complaint type: {short_name(tasks['complaint_type']['selected'])}", c_sel, tasks["complaint_type"]["candidates"][tasks["complaint_type"]["selected"]]["cv"]["f1_macro"]),
            row(f"Priority baseline: {short_name(base_name)}", base, tasks["priority"]["candidates"][base_name]["cv"]["f1_macro"]),
            row("Priority: Keras ANN", ann),
        ]
        md('<p class="section-sub" style="margin-top:1rem">Test-set metrics (macro-averaged) and 5-fold cross-validated macro-F1 on the training data.</p>'
           + table_html(["Model", "Accuracy", "Macro precision", "Macro recall", "Macro F1", "CV macro-F1"], t_rows))

        bs = tasks["priority"]["comparison"]["bootstrap"]["difference"]
        seeds = tasks["priority"]["ann"]["seed_summary"]
        includes0 = bs["ci95"][0] <= 0 <= bs["ci95"][1]
        verdict = ("The difference is small and its 95% interval includes zero, so the data do not show a reliable winner."
                   if includes0 else ("The ANN is slightly but reliably more accurate on this test set." if bs["mean"] > 0
                                      else "The traditional baseline is more accurate on this test set."))
        md('<div class="card info-card" style="margin-top:1rem"><h4>' + icon("network") + 'Keras ANN compared with the traditional baseline</h4>'
           f'<p>On the test set the ANN reaches {ann["accuracy"]:.1%} accuracy (macro-F1 {ann["macro_f1"]:.3f}) and the baseline '
           f'{esc(short_name(base_name))} reaches {base["accuracy"]:.1%} (macro-F1 {base["macro_f1"]:.3f}). '
           f'Across {len(tasks["priority"]["ann"]["seed_runs"])} training seeds the ANN averages {seeds["accuracy_mean"]:.1%} accuracy and '
           f'{seeds["macro_f1_mean"]:.3f} macro-F1. Accuracy difference (ANN minus baseline): {bs["mean"]:+.3f}, 95% bootstrap interval '
           f'[{bs["ci95"][0]:+.3f}, {bs["ci95"][1]:+.3f}]. {esc(verdict)}</p>'
           '<p>Priority is much harder than the other two tasks: the same message text appears with different priorities in the data, '
           'so no model can be certain from the text alone.</p></div>')

        h = tasks["priority"]["ann"]["history_deployed"]
        ep = list(range(1, len(h["loss"]) + 1))
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=ep, y=h["loss"], name="Training", mode="lines+markers", line=dict(color=cfg.COLORS["accent"], width=2.5))
            fig.add_scatter(x=ep, y=h["val_loss"], name="Validation", mode="lines+markers", line=dict(color=cfg.COLORS["high"], width=2.5))
            fig.add_vline(x=h["best_epoch"], line_dash="dash", line_color="#64748b", annotation_text="best epoch", annotation_font_color="#a3acd0")
            fig.update_xaxes(title="Epoch")
            show(style_fig(fig, "ANN loss", height=330, legend=True), "m_loss")
        with c2:
            fig = go.Figure()
            fig.add_scatter(x=ep, y=h["accuracy"], name="Training", mode="lines+markers", line=dict(color=cfg.COLORS["accent"], width=2.5))
            fig.add_scatter(x=ep, y=h["val_accuracy"], name="Validation", mode="lines+markers", line=dict(color=cfg.COLORS["high"], width=2.5))
            fig.add_vline(x=h["best_epoch"], line_dash="dash", line_color="#64748b")
            fig.update_xaxes(title="Epoch")
            show(style_fig(fig, "ANN accuracy", height=330, legend=True), "m_acc")

        with st.expander("Per-class report"):
            pick = st.selectbox("Model", ["Sentiment", "Complaint type", "Priority baseline", "Priority ANN"], key="perclass_pick")
            ev = {"Sentiment": s_sel, "Complaint type": c_sel, "Priority baseline": base, "Priority ANN": ann}[pick]
            md(table_html(["Class", "Precision", "Recall", "F1", "Test records"],
                          [[k, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", f"{v['support']:,}"] for k, v in ev["per_class"].items()]))

    with t_cm:
        md('<p class="section-sub">Rows are the true label, columns the predicted label. Colour shows the share of each true class; numbers are test records.</p>')
        items = [("Sentiment", s_sel, "cm_sent"), ("Complaint type", c_sel, "cm_comp"),
                 ("Priority: Keras ANN", ann, "cm_ann"), (f"Priority baseline: {short_name(base_name)}", base, "cm_base")]
        for i in range(0, 4, 2):
            cols = st.columns(2)
            for col, (title, ev, key) in zip(cols, items[i:i + 2]):
                cm = ev["confusion_matrix"]
                z = [[v / max(sum(r), 1) for v in r] for r in cm]
                with col:
                    heatmap(z, ev["labels"], ev["labels"], cm, title, key, height=380 if len(ev["labels"]) > 4 else 330)


# --------------------------------------------------------------------------- model information + footer
def render_model_info(results: Optional[dict], eda: Optional[dict]) -> None:
    sizes = (eda or {}).get("split_sizes", {})
    split_txt = (f"The data are split by distinct message: {sizes.get('train', 0):,} training rows and {sizes.get('test', 0):,} test rows, "
                 "and no message text appears on both sides. Otherwise the test score would only measure memorization.") if sizes else \
        "The data are split by distinct message so that no text appears in both training and test sets."
    cards = [
        ("filter", "NLP preprocessing", "Text is lower-cased, contractions are expanded, and punctuation and emoji are removed. Common filler words are dropped, "
         "but negations such as <code>not</code> and <code>never</code> are kept because they change the meaning."),
        ("type", "TF-IDF features", "TF-IDF turns each message into numbers. A word scores higher when it is frequent in the message but rare across all messages. "
         "Single words and two-word phrases are used."),
        ("layers", "Traditional classifiers", "Sentiment and complaint type use Logistic Regression, tuned with grid search. "
         "Naive Bayes and a linear SVM were trained for comparison, and a majority-class model shows what guessing achieves."),
        ("network", "Keras neural network", "Priority uses a feed-forward network: 256 input features, hidden layers of 128 and 64 ReLU units with dropout, and a 3-way softmax output. "
         "It learns by comparing its output with the true label (cross-entropy loss), sending the error backwards (backpropagation), and updating weights with the Adam optimizer. "
         "Training stops early when validation loss stops improving."),
        ("target", "Evaluation metrics", "Accuracy is the share of correct predictions. Precision asks how many predicted labels were right, recall asks how many true cases were found, "
         "and F1 combines both. Macro averages treat every class equally, which matters for small categories. A confusion matrix shows which classes get mixed up."),
        ("split", "Honest testing", split_txt + " Model choices use cross-validation on the training part only."),
    ]
    md('<span class="anchor" id="models"></span><h2 class="section-title">How it works</h2>'
       '<p class="section-sub">A short guide to the pipeline behind each prediction.</p><div class="info-grid">'
       + "".join(f'<div class="card info-card"><h4>{icon(i)}{esc(t)}</h4><p>{b}</p></div>' for i, t, b in cards) + '</div>')


def render_footer() -> None:
    md('<footer class="site-footer"><div><b>Customer Feedback Analyzer</b></div>'
       '<div>An AI capstone project that combines machine learning, NLP and a neural network to analyze customer feedback.</div>'
       '<div>For educational use only. Predictions are statistical estimates learned from a historical dataset and should not be the sole basis for business or customer decisions.</div></footer>')


# --------------------------------------------------------------------------- main
def main() -> None:
    inject_css()
    _init_state()
    predictor, load_error = get_predictor()
    meta = predictor.metadata if predictor else None
    results = read_report(cfg.RESULTS_JSON_PATH)
    eda = read_report(cfg.EDA_JSON_PATH)

    render_header(predictor is not None)
    render_hero(predictor, eda)
    render_analyzer(predictor, meta, load_error)
    render_analytics(results, eda)
    render_model_info(results, eda)
    render_footer()


main()


