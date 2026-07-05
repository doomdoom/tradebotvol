"""Self-contained HTML dashboard generator (light fintech theme).

Reads the prediction log from :class:`PredictionStorage` and renders a single
standalone page (inline CSS + inline JS, no external assets) so it works fully
offline and over the simple stdlib server.

Presentation only: it never changes how predictions are made, stored, scored,
or refreshed. Data is grouped **by base coin** (BTCUSDT -> BTC) purely for
display; the raw prediction log is untouched.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .evaluator import MIN_SAMPLE_FOR_RANKING, build_report
from .storage import PredictionStorage
from .utils import (
    BEARISH,
    BULLISH,
    DIRECTIONS,
    NEUTRAL,
    TIMEFRAME_MINUTES,
    display_tz_label,
    expected_price_range,
    format_price,
    to_display_time,
)

#: systemd unit that runs the prediction loop (see deploy/setup.sh).
PREDICTOR_SERVICE = "preduct-predictor.service"

_VALID_SAMPLE = 40  # resolved count above which a group is a "valid sample"
_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USD")
_COIN_COLORS = {
    "BTC": "#f7931a", "ETH": "#627eea", "SOL": "#9945ff", "BNB": "#f3ba2f",
    "XRP": "#23292f", "ADA": "#0033ad", "DOGE": "#c2a633", "AVAX": "#e84142",
}


# ---------------------------------------------------------------------- #
# Control-action logic (UNCHANGED — not presentation)
# ---------------------------------------------------------------------- #


def _predictor_status() -> str:
    """'active', 'inactive', or 'unknown' (e.g. off the VM / no systemd)."""
    if shutil.which("systemctl") is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["systemctl", "is-active", PREDICTOR_SERVICE],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_control_action(action: str, db_path: str, csv_path: str) -> str:
    """Execute a dashboard control action. Returns a short status message."""
    try:
        if action == "pause":
            subprocess.run(
                ["sudo", "systemctl", "stop", PREDICTOR_SERVICE], timeout=20, check=True
            )
            return "predictions paused"
        if action == "resume":
            subprocess.run(
                ["sudo", "systemctl", "start", PREDICTOR_SERVICE], timeout=20, check=True
            )
            return "predictions resumed"
        if action == "restart":
            subprocess.run(
                ["sudo", "systemctl", "restart", PREDICTOR_SERVICE], timeout=30, check=True
            )
            return "predictor restarted"
        if action == "clear":
            storage = PredictionStorage(
                db_path, csv_path, sqlite_enabled=True, csv_enabled=True
            )
            try:
                with storage._lock:
                    storage._conn.execute("DELETE FROM predictions")
                    storage._conn.commit()
                storage._export_csv()
            finally:
                storage.close()
            return "prediction log cleared"
        if action == "shutdown":
            # Guest-initiated poweroff stops the GCP instance and pauses
            # compute billing. Restart from the Cloud console.
            subprocess.Popen(["sudo", "shutdown", "-h", "now"])
            return "shutting down"
    except Exception as exc:
        return f"action failed: {exc}"
    return "unknown action"


def apply_settings(config_path: str, form: dict) -> str:
    """Validate + persist a few config.json fields, then restart the predictor."""
    from .config import Config

    try:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if form.get("symbols"):
            syms = [s.strip().upper() for s in form["symbols"].split(",") if s.strip()]
            if syms:
                raw["symbols"] = syms
        if form.get("timeframes"):
            tfs = [t.strip() for t in form["timeframes"].split(",") if t.strip()]
            if tfs:
                raw["timeframes"] = tfs
        if form.get("neutral_threshold_pct"):
            raw["neutral_threshold_pct"] = float(form["neutral_threshold_pct"])
        if form.get("min_confidence"):
            raw["min_confidence"] = float(form["min_confidence"])
        Config(**raw)  # validates symbols/timeframes/ranges; raises on bad input
        Path(config_path).write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        if shutil.which("systemctl") is not None:
            subprocess.run(
                ["sudo", "systemctl", "restart", PREDICTOR_SERVICE], timeout=30, check=False
            )
        return "settings saved — predictor restarting with the new configuration"
    except Exception as exc:
        return f"settings not saved: {exc}"


# ---------------------------------------------------------------------- #
# Formatting + small helpers
# ---------------------------------------------------------------------- #


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def _int(value: float) -> str:
    return f"{int(value):,}"


def _base_coin(symbol: str) -> str:
    s = str(symbol).upper()
    for q in _QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def _coin_color(coin: str) -> str:
    return _COIN_COLORS.get(coin.upper(), "#475569")


def _accuracy_tone(acc: float, resolved: int) -> tuple[str, str, str]:
    """(tone, status_label, sample_label) for an accuracy given its sample size."""
    if resolved < MIN_SAMPLE_FOR_RANKING:
        sample = "Insufficient data" if resolved == 0 else "Low sample"
        return "info", "Insufficient data", sample
    sample = "Valid sample" if resolved >= _VALID_SAMPLE else "Low sample"
    if acc >= 50.0:
        return "good", "Strong", sample
    if acc >= 40.0:
        return "warn", "Weak", sample
    return "bad", "Weak", sample


def _badge(text: str, tone: str = "neutral", title: str = "") -> str:
    t = f' title="{_esc(title)}"' if title else ""
    return f'<span class="badge badge-{tone}"{t}>{_esc(text)}</span>'


ICONS = {
    "total": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5"/>',
    "resolved": '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
    "pending": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "accuracy": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>',
    "pairs": ('<rect x="3" y="3" width="7" height="7" rx="1.6"/>'
              '<rect x="14" y="3" width="7" height="7" rx="1.6"/>'
              '<rect x="3" y="14" width="7" height="7" rx="1.6"/>'
              '<rect x="14" y="14" width="7" height="7" rx="1.6"/>'),
    "dash": ('<rect x="3" y="3" width="8" height="8" rx="2"/>'
             '<rect x="13" y="3" width="8" height="5" rx="2"/>'
             '<rect x="13" y="10" width="8" height="11" rx="2"/>'
             '<rect x="3" y="13" width="8" height="8" rx="2"/>'),
    "coins": '<circle cx="8" cy="8" r="5"/><circle cx="15" cy="15" r="5"/>',
    "log": '<path d="M4 6h16M4 12h16M4 18h10"/>',
    "gear": ('<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3'
             'M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>'),
    "shield": '<path d="M12 3l7 3v6c0 4-3 7-7 9-4-2-7-5-7-9V6l7-3z"/><path d="M9 12l2 2 4-4"/>',
}


def _icon(name: str, size: int = 20) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{ICONS.get(name, "")}</svg>'
    )


def _prob_bar(bull: float, bear: float, neutral: float, width: int = 120) -> str:
    segs = [(BULLISH, bull, "#16a34a"), (NEUTRAL, neutral, "#94a3b8"), (BEARISH, bear, "#dc2626")]
    x = 0.0
    rects = []
    for name, frac, color in segs:
        w = max(0.0, frac) * width
        if w > 0:
            rects.append(
                f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="9" rx="1" fill="{color}">'
                f"<title>{name}: {frac * 100:.0f}%</title></rect>"
            )
            x += w
    return (
        f'<svg width="{width}" height="9" viewBox="0 0 {width} 9" role="img" '
        f'aria-label="probabilities" style="border-radius:3px;background:#eef2f7">'
        f'{"".join(rects)}</svg>'
    )


def _acc_bar(acc: float, tone: str, height: int = 8) -> str:
    w = max(2.0, min(100.0, acc))
    return (
        f'<span class="bar"><span class="bar-fill" '
        f'style="width:{w:.1f}%;background:var(--{tone})"></span></span>'
    )


# ---------------------------------------------------------------------- #
# Coin grouping (UI only — derived from the existing prediction log)
# ---------------------------------------------------------------------- #


def _coin_groups(all_df: pd.DataFrame, report: dict) -> list[dict]:
    """Group the prediction log by base coin for display (BTCUSDT -> BTC)."""
    if all_df.empty:
        return []
    gr = {(g.symbol, g.timeframe): g for g in report.get("groups", [])}
    df = all_df.copy()
    df["_coin"] = df["symbol"].map(_base_coin)
    df["_resolved"] = df["actual_direction"].notna()

    coins: list[dict] = []
    for coin, cdf in df.groupby("_coin"):
        rdf = cdf[cdf["_resolved"]]
        resolved_n = len(rdf)
        acc = float(rdf["prediction_correct"].fillna(0).mean() * 100) if resolved_n else 0.0

        tf_entries = []
        for (sym, tf), tdf in cdf.groupby(["symbol", "timeframe"]):
            tr = tdf[tdf["_resolved"]]
            t_res = len(tr)
            t_acc = float(tr["prediction_correct"].fillna(0).mean() * 100) if t_res else 0.0
            tf_entries.append(
                {
                    "symbol": sym, "timeframe": tf, "resolved": t_res,
                    "pending": len(tdf) - t_res, "accuracy": t_acc,
                    "report": gr.get((sym, tf)),
                }
            )
        tf_entries.sort(key=lambda e: TIMEFRAME_MINUTES.get(e["timeframe"], 9999))
        ranked = [e for e in tf_entries if e["resolved"] >= MIN_SAMPLE_FOR_RANKING]
        best = max(ranked, key=lambda e: e["accuracy"]) if ranked else None
        worst = min(ranked, key=lambda e: e["accuracy"]) if ranked else None

        coins.append(
            {
                "coin": coin,
                "symbols": sorted(cdf["symbol"].unique().tolist()),
                "total": len(cdf), "resolved": resolved_n,
                "pending": len(cdf) - resolved_n, "accuracy": acc,
                "timeframes": tf_entries, "best": best, "worst": worst,
                "recent": cdf.sort_values("id", ascending=False).head(12),
            }
        )
    coins.sort(key=lambda c: (-c["total"], c["coin"]))
    return coins


# ---------------------------------------------------------------------- #
# Components
# ---------------------------------------------------------------------- #


def _sidebar() -> str:
    links = [
        ("overview", "dash", "Overview"),
        ("coins", "coins", "Coins"),
        ("log", "log", "Recent"),
        ("controls", "gear", "Controls"),
    ]
    items = "".join(
        f'<a class="rail-item" href="#{anchor}" title="{label}" aria-label="{label}">'
        f"{_icon(icon)}</a>"
        for anchor, icon, label in links
    )
    return f"""
<aside class="rail" aria-label="Sections">
  <div class="rail-logo" aria-hidden="true">◈</div>
  {items}
</aside>"""


def _header(generated: str, refresh_seconds: int | None, tz: str) -> str:
    live = ""
    if refresh_seconds and refresh_seconds > 0:
        live = (
            '<span class="chip chip-live" id="live-chip"><span class="dot"></span>'
            '<span id="refresh-status">Live</span></span>'
        )
    return f"""
<header class="topbar">
  <div>
    <h1>Next-Candle Prediction Dashboard</h1>
    <p class="subtitle">Research dashboard for validating model prediction accuracy
    across symbols and timeframes.</p>
  </div>
  <div class="chips">
    {live}
    <span class="chip chip-sel" id="sel-chip" hidden>—</span>
    <span class="chip">Source: SQLite / CSV prediction log</span>
    <span class="chip chip-muted" id="last-updated">Updated {_esc(generated)} {_esc(tz)}</span>
    <button class="chip chip-btn" id="settings-jump" type="button" hidden>⚙ Settings</button>
  </div>
</header>"""


def _verdict(resolved: pd.DataFrame, report: dict) -> str:
    n = len(resolved)
    overall = report.get("overall_accuracy_pct", 0.0)
    if n == 0:
        tone, icon, head, sub = (
            "info", "⏳", "Just getting started",
            "No predictions have been scored yet. Each prediction is checked when its "
            "candle closes — check back in a few minutes.",
        )
    elif n < MIN_SAMPLE_FOR_RANKING:
        tone, icon, head, sub = (
            "warn", "📈", f"Collecting data — {n} of {MIN_SAMPLE_FOR_RANKING} scored",
            "Too early to judge. Accuracy over a handful of predictions is just noise.",
        )
    elif overall >= 50:
        tone, icon, head, sub = (
            "good", "✅", f"Showing an edge — {_pct(overall)} accuracy",
            f"Across {_int(n)} scored predictions the model beats a coin flip. Keep validating.",
        )
    elif overall >= 40:
        tone, icon, head, sub = (
            "warn", "⚠️", f"No clear edge yet — {_pct(overall)} accuracy",
            f"Across {_int(n)} scored predictions, accuracy is near the ~33% random baseline.",
        )
    else:
        tone, icon, head, sub = (
            "bad", "🔻", f"No edge — {_pct(overall)} accuracy",
            f"Across {_int(n)} scored predictions, accuracy is at or below random. Not reliable.",
        )
    return f"""
<section class="verdict" id="sec-verdict" data-tone="{tone}" aria-label="Current status">
  <div class="verdict-icon" aria-hidden="true">{icon}</div>
  <div><div class="verdict-head">{_esc(head)}</div>
    <div class="verdict-sub">{_esc(sub)}</div></div>
</section>"""


def _control_panel(controls_enabled: bool, message: str = "") -> str:
    if not controls_enabled:
        return ""
    status = _predictor_status()
    running = status == "active"
    if status == "unknown":
        chip = _badge("status unknown", "info")
    else:
        chip = _badge("running" if running else "paused", "good" if running else "bad")
    toggle_action = "pause" if running else "resume"
    toggle_label = "Pause predictions" if running else "Resume predictions"

    def form(action: str, label: str, tone: str, confirm: str = "") -> str:
        data = f' data-confirm="{_esc(confirm)}"' if confirm else ""
        return (
            f"<form method='post' action='/action' class='ctl-form'{data}>"
            f"<input type='hidden' name='action' value='{action}'>"
            f"<button class='btn btn-{tone}' type='submit'>{_esc(label)}</button></form>"
        )

    note = f'<span class="ctl-msg">{_esc(message)}</span>' if message else ""
    return f"""
<section class="card ctl-panel" id="controls" aria-label="Controls">
  <div class="ctl-status">
    <span class="card-title">Controls</span>
    <span class="muted">predictor</span> {chip}</div>
  <div class="ctl-actions">
    {form(toggle_action, toggle_label, "primary")}
    {form("restart", "Restart", "ghost", "Restart the predictor? It reloads the latest settings.")}
    <a class="btn btn-ghost" href="/" title="Reload now">↻ Refresh</a>
    {form("clear", "Clear log", "warn", "Clear ALL predictions? This cannot be undone.")}
    {form("shutdown", "Shut down", "danger",
          "Shut down the VM? Billing pauses. Restart it from the Google Cloud console.")}
  </div>
  {note}
</section>"""


def _settings_panel(controls_enabled: bool, config_path: str | None) -> str:
    if not controls_enabled or not config_path:
        return ""
    try:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return ""
    syms = _esc(", ".join(raw.get("symbols", [])))
    tfs = _esc(", ".join(raw.get("timeframes", [])))
    nt = _esc(raw.get("neutral_threshold_pct", 0.03))
    mc = _esc(raw.get("min_confidence", 0.55))
    return f"""
<details class="card disc" id="settings-panel">
  <summary><span class="card-title">⚙ Settings</span>
    <span class="muted">change what the bot predicts — saves &amp; restarts</span></summary>
  <form method="post" action="/settings" class="settings-form"
        data-confirm="Save settings and restart the predictor?">
    <label>Symbols <span class="muted">(e.g. BTCUSDT, ETHUSDT)</span>
      <input name="symbols" value="{syms}"></label>
    <label>Timeframes <span class="muted">(e.g. 1m, 5m, 15m)</span>
      <input name="timeframes" value="{tfs}"></label>
    <label>Neutral threshold %
      <input name="neutral_threshold_pct" type="number" step="0.005" value="{nt}"></label>
    <label>Min confidence
      <input name="min_confidence" type="number" step="0.05" value="{mc}"></label>
    <button class="btn btn-primary" type="submit">Save &amp; restart</button>
  </form>
</details>"""


def _how_to_read() -> str:
    return """
<details class="card disc">
  <summary><span class="card-title">❓ How to read this dashboard</span></summary>
  <div class="how-body"><ul>
    <li><b>It predicts the next candle</b> — up (bullish), down (bearish) or little
    change (neutral) — then scores itself once that candle closes.</li>
    <li><b>Accuracy</b> is how often it was right. For a 3-way guess ~33% is random;
    above ~50% would be a real edge.</li>
    <li><b>Resolved</b> = already scored. <b>Pending</b> = waiting for the candle to
    close. Rankings and Strong/Weak labels stay hidden below
    """ + str(MIN_SAMPLE_FOR_RANKING) + """ resolved so you don't read noise as signal.</li>
    <li><b>Green</b> = strong, <b>orange/red</b> = weak, <b>blue</b> = not enough data.</li>
    <li>Research tool — it never trades. Focus on higher timeframes (15m, 1h).</li>
  </ul></div>
</details>"""


def _kpi(value: str, label: str, helper: str, tone: str, icon: str, badge: str = "") -> str:
    badge_html = f'<div class="kpi-badge">{badge}</div>' if badge else ""
    return f"""
    <div class="kpi" data-tone="{tone}">
      <div class="kpi-top">
        <span class="kpi-ico" data-tone="{tone}">{_icon(icon, 18)}</span>
        <span class="kpi-label">{_esc(label)}</span>{badge_html}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-help">{helper}</div>
    </div>"""


def _kpis(all_df: pd.DataFrame, resolved: pd.DataFrame, report: dict) -> str:
    total = len(all_df)
    n_resolved = len(resolved)
    pending = total - n_resolved
    n_pairs = all_df.groupby(["symbol", "timeframe"]).ngroups if total else 0
    overall = report.get("overall_accuracy_pct", 0.0)

    if n_resolved == 0:
        acc_tone, acc_help, acc_badge, acc_val = "info", "Awaiting resolved predictions", "", "—"
    else:
        acc_tone = "good" if overall >= 50 else ("warn" if overall >= 40 else "bad")
        acc_val = _pct(overall)
        if overall < 50:
            acc_help = f"Below baseline · over {_int(n_resolved)} resolved"
            acc_badge = _badge("Weak performance", acc_tone)
        else:
            acc_help = f"Above baseline · over {_int(n_resolved)} resolved"
            acc_badge = _badge("Edge", acc_tone)

    cards = [
        _kpi(_int(total), "Total predictions", "All predictions logged", "info", "total"),
        _kpi(_int(n_resolved), "Resolved", "Scored against the candle", "info", "resolved"),
        _kpi(_int(pending), "Pending", "Awaiting candle close",
             "warn" if pending else "info", "pending"),
        _kpi(acc_val, "Overall accuracy", acc_help, acc_tone, "accuracy", acc_badge),
        _kpi(_int(n_pairs), "Markets tracked", "Symbol × timeframe", "info", "pairs"),
    ]
    return f'<div class="kpi-grid" id="overview">{"".join(cards)}</div>'


def _validation_health(all_df: pd.DataFrame, resolved: pd.DataFrame, report: dict) -> str:
    n_resolved = len(resolved)
    pending = len(all_df) - n_resolved
    groups_ready = sum(
        1 for g in report.get("groups", []) if g.total_predictions >= MIN_SAMPLE_FOR_RANKING
    )
    coins_ready = len({
        _base_coin(g.symbol)
        for g in report.get("groups", [])
        if g.total_predictions >= MIN_SAMPLE_FOR_RANKING
    })
    overall = report.get("overall_accuracy_pct", 0.0)

    if n_resolved == 0:
        tone, label, note = "info", "Awaiting data", "No predictions have resolved yet."
    elif n_resolved < MIN_SAMPLE_FOR_RANKING:
        tone, label, note = ("warn", "Insufficient sample size",
            f"Only {n_resolved} resolved — rankings unlock at {MIN_SAMPLE_FOR_RANKING}+ per group.")
    elif groups_ready == 0:
        tone, label, note = ("info", "Validation in progress",
            f"{n_resolved} resolved, but no single market has {MIN_SAMPLE_FOR_RANKING}+ yet.")
    elif overall < 40:
        tone, label, note = ("bad", "Statistically weak",
            "Accuracy is near or below the random baseline. No usable edge so far.")
    else:
        tone, label, note = ("good", "Ready for deeper review",
            "Enough data to start judging performance per market.")

    stats = [
        ("Resolved sample", _int(n_resolved)),
        ("Pending", _int(pending)),
        ("Min. per group", _int(MIN_SAMPLE_FOR_RANKING)),
        ("Coins ready", _int(coins_ready)),
    ]
    stat_html = "".join(
        f'<div class="hstat"><div class="hstat-v">{v}</div>'
        f'<div class="hstat-l">{_esc(k)}</div></div>'
        for k, v in stats
    )
    return f"""
<section class="card" id="sec-validation" aria-label="Validation health">
  <div class="card-head"><h2>Validation health</h2>{_badge(label, tone)}</div>
  <p class="muted card-note">{_esc(note)} Rankings and Strong/Weak labels stay hidden
  for any group below {MIN_SAMPLE_FOR_RANKING} resolved predictions.</p>
  <div class="hstats">{stat_html}</div>
</section>"""


def _rankings(report: dict) -> str:
    def mini(label: str, entry: tuple | None) -> str:
        if not entry:
            return (f'<div class="rank-card"><div class="rank-l">{_esc(label)}</div>'
                    f'<div class="rank-empty">{_badge("not enough data", "info")}</div></div>')
        name, acc, n = entry
        tone, _s, _q = _accuracy_tone(acc, n)
        return (
            f'<div class="rank-card"><div class="rank-l">{_esc(label)}</div>'
            f'<div class="rank-v">{_esc(name)}</div>'
            f'<div class="rank-a" style="color:var(--{tone})">{_pct(acc)}</div>'
            f'<div class="muted">{_int(n)} resolved</div></div>'
        )

    return f"""
<section class="card" id="sec-rankings" aria-label="Model rankings">
  <div class="card-head"><h2>Model rankings</h2></div>
  <p class="muted card-note">Shown only once a group has at least
  {MIN_SAMPLE_FOR_RANKING} resolved predictions.</p>
  <div class="rank-grid">
    {mini("Best timeframe", report.get("best_timeframe"))}
    {mini("Worst timeframe", report.get("worst_timeframe"))}
    {mini("Best coin", report.get("best_symbol"))}
    {mini("Worst coin", report.get("worst_symbol"))}
  </div>
</section>"""


def _confusion_table(confusion: pd.DataFrame) -> str:
    total = int(confusion.to_numpy().sum()) or 1
    head = "".join(f"<th scope='col'>{_esc(c)}</th>" for c in DIRECTIONS)
    body = []
    for pred in DIRECTIONS:
        cells = []
        for actual in DIRECTIONS:
            v = int(confusion.loc[pred, actual])
            intensity = v / total
            if pred == actual:
                bg = f"rgba(22,163,74,{0.08 + intensity:.2f})"
            else:
                bg = f"rgba(148,163,184,{0.05 + intensity * 0.4:.2f})"
            cells.append(f'<td style="background:{bg}">{v}</td>')
        body.append(f"<tr><th scope='row'>{_esc(pred)}</th>{''.join(cells)}</tr>")
    return (
        "<table class='confusion'><thead><tr><th scope='col'>pred \\ actual</th>"
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _timeframe_card(e: dict) -> str:
    tone, status, sample = _accuracy_tone(e["accuracy"], e["resolved"])
    acc_txt = _pct(e["accuracy"]) if e["resolved"] else "—"
    return f"""
    <div class="tf-card" data-tone="{tone}">
      <div class="tf-top"><span class="tf-name">{_esc(e['timeframe'])}</span>
        {_badge(status, tone)}</div>
      <div class="tf-acc" style="color:var(--{tone})">{acc_txt}</div>
      {_acc_bar(e['accuracy'], tone)}
      <div class="tf-meta">
        <span>{_int(e['resolved'])} resolved</span>
        <span class="muted">{_int(e['pending'])} pending</span>
      </div>
      <div class="tf-sample">{_badge(sample, tone)}</div>
    </div>"""


def _prediction_rows(df: pd.DataFrame, offset_hours: float) -> str:
    rows = []
    for r in df.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else dict(zip(df.columns, r))
        pred = str(d["predicted_direction"])
        coin = _base_coin(d["symbol"])
        target = to_display_time(d.get("target_candle_time", ""), offset_hours)
        prob = _prob_bar(
            float(d["bullish_probability"]), float(d["bearish_probability"]),
            float(d["neutral_probability"]),
        )
        conf = float(d.get("confidence") or 0.0)
        conf_label = str(d.get("confidence_label", "") or "")
        ref = float(d.get("reference_close") or 0.0)
        mn, mx = d.get("expected_move_min_pct"), d.get("expected_move_max_pct")
        if ref > 0 and mn is not None and mx is not None and not pd.isna(mn):
            low, high = expected_price_range(ref, float(mn), float(mx))
            price = (f"{format_price(ref)}<span class='muted'> → "
                     f"{format_price(low)}–{format_price(high)}</span>")
        else:
            price = "<span class='muted'>—</span>"
        pred_badge = _badge(
            pred, "good" if pred == BULLISH else "bad" if pred == BEARISH else "neutral"
        )
        actual = d.get("actual_direction")
        if actual is None or (isinstance(actual, float) and pd.isna(actual)):
            actual_cell = "<span class='muted'>—</span>"
            result = _badge("pending", "warn")
            result_key = "pending"
        else:
            a = str(actual)
            actual_cell = _badge(
                a, "good" if a == BULLISH else "bad" if a == BEARISH else "neutral"
            )
            correct = int(d.get("prediction_correct") or 0) == 1
            result = _badge("correct", "good") if correct else _badge("incorrect", "bad")
            result_key = "correct" if correct else "incorrect"
        rows.append(
            f'<tr data-row data-coin="{_esc(coin)}" data-tf="{_esc(d["timeframe"])}" '
            f'data-result="{result_key}">'
            f'<td class="nowrap muted">{_esc(target)}</td>'
            f'<td class="nowrap"><b>{_esc(coin)}</b></td>'
            f'<td>{_badge(_esc(d["timeframe"]), "plain")}</td>'
            f"<td>{pred_badge}</td>"
            f'<td class="hide-sm">{prob}</td>'
            f'<td class="nowrap">{conf_label} <span class="muted">{conf * 100:.0f}%</span></td>'
            f'<td class="nowrap hide-sm">{price}</td>'
            f"<td>{actual_cell}</td><td>{result}</td></tr>"
        )
    return "".join(rows)


_LOG_HEAD = (
    "<thead><tr>"
    '<th scope="col">Target candle</th><th scope="col">Coin</th>'
    '<th scope="col">TF</th><th scope="col">Predicted</th>'
    '<th scope="col" class="hide-sm">Probability</th>'
    '<th scope="col">Confidence</th>'
    '<th scope="col" class="hide-sm">Expected close</th>'
    '<th scope="col">Actual</th><th scope="col">Result</th>'
    "</tr></thead>"
)


def _coin_accordion(coins: list[dict], offset_hours: float) -> str:
    if not coins:
        return """
<section class="card" id="coins" aria-label="Performance by coin">
  <div class="card-head"><h2>Performance by coin</h2></div>
  <div class="empty">Waiting for predictions…</div>
</section>"""

    blocks = []
    for i, c in enumerate(coins):
        color = _coin_color(c["coin"])
        tone, status, _sample = _accuracy_tone(c["accuracy"], c["resolved"])
        acc_txt = _pct(c["accuracy"]) if c["resolved"] else "—"
        pair = ", ".join(c["symbols"])
        best = (f'{c["best"]["timeframe"]} · {_pct(c["best"]["accuracy"])}'
                if c["best"] else "—")
        worst = (f'{c["worst"]["timeframe"]} · {_pct(c["worst"]["accuracy"])}'
                 if c["worst"] else "—")
        tf_cards = "".join(_timeframe_card(e) for e in c["timeframes"])

        recent = c["recent"]
        if recent.empty:
            recent_html = f'<div class="empty">No {_esc(c["coin"])} predictions yet</div>'
        else:
            recent_html = (
                f'<div class="table-scroll"><table class="log">{_LOG_HEAD}'
                f'<tbody>{_prediction_rows(recent, offset_hours)}</tbody></table></div>'
            )

        detail_blocks = [
            f"<div class='pd'><div class='pd-h'>{_esc(e['symbol'])} "
            f"{_esc(e['timeframe'])}</div>{_confusion_table(e['report'].confusion)}</div>"
            for e in c["timeframes"] if e["report"] is not None
        ]
        detail_html = (
            f"<details class='disc sub'><summary><span class='card-title'>"
            f"Per-timeframe detail</span></summary><div class='pd-grid'>"
            f"{''.join(detail_blocks)}</div></details>"
            if detail_blocks else ""
        )

        open_attr = " open" if i == 0 else ""
        blocks.append(
            f"""
  <details class="coin" name="coin"{open_attr}>
    <summary>
      <span class="coin-ava" style="background:{color}">{_esc(c['coin'][:3])}</span>
      <span class="coin-id">
        <span class="coin-sym">{_esc(c['coin'])}</span>
        <span class="muted">{_esc(pair)}</span></span>
      <span class="coin-acc" style="color:var(--{tone})">{acc_txt}</span>
      {_badge(status, tone)}
      <span class="coin-bar">{_acc_bar(c['accuracy'], tone)}</span>
      <span class="coin-counts muted">{_int(c['resolved'])} resolved ·
        {_int(c['pending'])} pending</span>
      <span class="chev" aria-hidden="true">⌄</span>
    </summary>
    <div class="coin-body">
      <div class="coin-kv">
        <span>Total <b>{_int(c['total'])}</b></span>
        <span>Resolved <b>{_int(c['resolved'])}</b></span>
        <span>Pending <b>{_int(c['pending'])}</b></span>
        <span>Best TF <b>{_esc(best)}</b></span>
        <span>Worst TF <b>{_esc(worst)}</b></span>
      </div>
      <h3 class="sub-h">Timeframe performance</h3>
      <div class="tf-grid">{tf_cards}</div>
      <h3 class="sub-h">Recent {_esc(c['coin'])} predictions</h3>
      {recent_html}
      {detail_html}
    </div>
  </details>"""
        )
    return f"""
<section id="coins" aria-label="Performance by coin">
  <div class="section-head"><h2>Selected coin performance</h2>
    <span class="muted">detailed timeframes &amp; predictions for the coin chosen above</span></div>
  <div class="coin-list">{''.join(blocks)}</div>
</section>"""


def _recent_log(all_df: pd.DataFrame, offset_hours: float) -> str:
    if all_df.empty:
        return """
<section class="card" id="log" aria-label="Recent predictions">
  <div class="card-head"><h2>Recent predictions</h2></div>
  <div class="empty">Waiting for predictions…</div>
</section>"""

    recent = all_df.sort_values("id", ascending=False).head(30)
    coins = sorted({_base_coin(s) for s in recent["symbol"].unique()})
    tfs = sorted({str(t) for t in recent["timeframe"].unique()},
                 key=lambda t: TIMEFRAME_MINUTES.get(t, 9999))

    def group(name: str, values: list[str], labels=None) -> str:
        labels = labels or {}
        btns = [f'<button class="fbtn active" data-f="{name}" data-v="all" type="button">All</button>']
        for v in values:
            btns.append(
                f'<button class="fbtn" data-f="{name}" data-v="{_esc(v)}" type="button">'
                f'{_esc(labels.get(v, v))}</button>'
            )
        return f'<div class="fgroup" data-group="{name}"><span class="flabel">{name.title()}</span>{"".join(btns)}</div>'

    filters = (
        group("coin", coins)
        + group("tf", tfs)
        + group("result", ["correct", "incorrect", "pending"])
    )
    return f"""
<section class="card" id="log" aria-label="Recent predictions">
  <div class="card-head"><h2>Recent predictions</h2>
    <span class="muted">latest {len(recent)}</span></div>
  <div class="filters">{filters}</div>
  <div class="table-scroll">
    <table class="log" id="log-table">{_LOG_HEAD}
      <tbody>{_prediction_rows(recent, offset_hours)}</tbody></table>
  </div>
  <div class="empty filter-empty" hidden>No predictions match these filters.</div>
</section>"""


def _modal() -> str:
    return """
<div class="modal" id="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title" hidden>
  <div class="modal-card">
    <div class="modal-title" id="modal-title">Please confirm</div>
    <div class="modal-msg" id="modal-msg"></div>
    <div class="modal-actions">
      <button class="btn btn-ghost" id="modal-cancel" type="button">Cancel</button>
      <button class="btn btn-danger" id="modal-ok" type="button">Confirm</button>
    </div>
  </div>
</div>"""


def _error_page(message: str, refresh_seconds: int | None) -> str:
    meta = (f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
            if refresh_seconds and refresh_seconds > 0 else "")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">{meta}
<title>Dashboard — data error</title><style>{_CSS}</style></head>
<body><main class="shell"><div class="content"><div class="notice notice-error">
<div class="notice-icon">⚠</div><div><div class="notice-title">Could not load prediction data</div>
<div class="notice-body">{_esc(message)}</div></div></div></div></main></body></html>"""


def _dashboard_data(all_df: pd.DataFrame, offset_hours: float) -> dict:
    """Serialize the latest per-(coin, timeframe) prediction plus a recent price
    series, for the top hero panel + chart. Reads the existing prediction log
    only — no prediction, model, scoring, Binance or storage logic is involved.
    """
    out: dict = {"coins": [], "tfs": {}, "pred": {}, "sym": {}}
    if all_df.empty:
        return out
    df = all_df.copy()
    df["_coin"] = df["symbol"].map(_base_coin)
    order = df.groupby("_coin").size().sort_values(ascending=False)
    out["coins"] = [str(c) for c in order.index.tolist()]
    for coin in out["coins"]:
        cdf = df[df["_coin"] == coin]
        out["sym"][coin] = str(cdf["symbol"].iloc[-1])  # BTC -> BTCUSDT
        tfs = sorted(
            cdf["timeframe"].unique().tolist(),
            key=lambda t: TIMEFRAME_MINUTES.get(str(t), 9999),
        )
        out["tfs"][coin] = [str(t) for t in tfs]
        out["pred"][coin] = {}
        for tf in tfs:
            tdf = cdf[cdf["timeframe"] == tf].sort_values("id")
            if tdf.empty:
                continue
            last = tdf.iloc[-1]
            pdir = str(last["predicted_direction"])
            ref = float(last["reference_close"]) if pd.notna(last.get("reference_close")) else 0.0
            mn, mx = last.get("expected_move_min_pct"), last.get("expected_move_max_pct")
            lo = hi = None
            if ref > 0 and mn is not None and mx is not None and pd.notna(mn) and pd.notna(mx):
                lo, hi = expected_price_range(ref, float(mn), float(mx))
            actual = last.get("actual_direction")
            resolved = pd.notna(actual)
            correct = (int(last.get("prediction_correct") or 0) == 1) if resolved else None
            series = [
                float(x) for x in tdf["reference_close"].tail(48).tolist()
                if pd.notna(x) and x > 0
            ]
            out["pred"][coin][str(tf)] = {
                "dir": pdir,
                "up": pdir == BULLISH,
                "down": pdir == BEARISH,
                "bull": round(float(last["bullish_probability"]), 4),
                "bear": round(float(last["bearish_probability"]), 4),
                "neu": round(float(last["neutral_probability"]), 4),
                "conf": round(float(last["confidence"]) if pd.notna(last.get("confidence")) else 0.0, 4),
                "conf_label": str(last.get("confidence_label") or ""),
                "ref": ref,
                "exp_low": (round(lo, 4) if lo is not None else None),
                "exp_high": (round(hi, 4) if hi is not None else None),
                "target": to_display_time(last.get("target_candle_time", ""), offset_hours),
                "status": "resolved" if resolved else "pending",
                "correct": correct,
                "actual": (str(actual) if resolved else None),
                "series": series,
            }
    return out


def _hero() -> str:
    """Static skeleton for the top prediction hero + chart. Structure is rendered
    once; JS only updates VALUES (text, bar widths, chart path attributes) so the
    hero never re-renders/flickers. Selectors are filled by JS from the data."""
    return """
<section class="card hero" id="sec-hero" aria-label="Next candle prediction">
  <div class="hero-head-row">
    <div>
      <div class="hero-kicker">NEXT CANDLE PREDICTION</div>
      <div class="hero-sel">
        <div class="seg" id="coin-seg" role="tablist" aria-label="Coin"></div>
        <div class="seg" id="tf-seg" role="tablist" aria-label="Timeframe"></div>
      </div>
    </div>
    <div class="hero-when muted">target candle <span id="hero-target">—</span></div>
  </div>
  <div class="hero-grid">
    <div class="hero-call" id="hero-call">
      <div class="hero-arrow" id="hero-arrow">–</div>
      <div>
        <div class="hero-verdict" id="hero-verdict">Waiting for data…</div>
        <div class="hero-meta muted" id="hero-meta">—</div>
        <div class="hero-status" id="hero-status"></div>
      </div>
    </div>
    <div class="hero-facts" id="hero-facts">
      <div class="fact"><div class="fact-l">Confidence</div>
        <div class="fact-v" id="f-conf">—</div></div>
      <div class="fact"><div class="fact-l">Probability</div>
        <div class="fact-v"><span class="pbar"><i id="pb-up"></i><i id="pb-neu"></i><i id="pb-down"></i></span>
        <div class="pkey muted" id="pb-key">—</div></div></div>
      <div class="fact"><div class="fact-l">Expected close</div>
        <div class="fact-v" id="f-exp">—</div></div>
      <div class="fact"><div class="fact-l">Last close</div>
        <div class="fact-v" id="f-last">—</div></div>
    </div>
    <div class="hero-chartwrap">
      <div class="hero-chart-top"><span id="chart-label">Recent price</span>
        <span id="chart-range"></span></div>
      <svg class="hero-svg" id="hero-svg" viewBox="0 0 600 150"
           preserveAspectRatio="none" aria-label="recent price chart">
        <defs><linearGradient id="cg" x1="0" x2="0" y1="0" y2="1">
          <stop id="cg0" offset="0" stop-color="#64748b" stop-opacity="0.16"/>
          <stop id="cg1" offset="1" stop-color="#64748b" stop-opacity="0"/></linearGradient></defs>
        <path id="ch-area" d="" fill="url(#cg)"></path>
        <path id="ch-line" d="" fill="none" stroke="#64748b" stroke-width="2"
              stroke-linejoin="round" stroke-linecap="round"></path>
        <circle id="ch-dot" r="3.5" fill="#64748b" cx="-10" cy="-10"></circle>
      </svg>
      <div class="empty" id="chart-empty" hidden>Market chart data unavailable.</div>
    </div>
  </div>
</section>"""


def _market_chart() -> str:
    """Static skeleton for the live candlestick chart. All values (candles,
    price, OHLC, countdown, signal) are drawn by JS from /api/market-candles."""
    return """
<section class="card mkt" id="sec-market" aria-label="Live market chart">
  <div class="mkt-head">
    <div>
      <div class="hero-kicker">LIVE MARKET</div>
      <div class="mkt-title"><span id="mkt-pair">—</span>
        <span class="mkt-state" id="mkt-state">connecting…</span></div>
    </div>
    <div class="mkt-price">
      <span class="mkt-price-v" id="mkt-price">—</span>
      <span class="mkt-dir badge badge-neutral" id="mkt-dir">—</span>
    </div>
  </div>
  <div class="mkt-grid">
    <div class="mkt-chartwrap">
      <svg class="mkt-svg" id="mkt-svg" viewBox="0 0 720 300"
           preserveAspectRatio="none" aria-label="candlestick chart">
        <g id="mkt-grid"></g><g id="mkt-plot"></g>
      </svg>
      <div class="mkt-signal" id="mkt-signal" hidden></div>
      <div class="empty mkt-empty" id="mkt-empty">Waiting for live candle data…</div>
    </div>
    <div class="mkt-panel">
      <div class="mkt-stat"><span>Open</span><b id="mkt-o">—</b></div>
      <div class="mkt-stat"><span>High</span><b id="mkt-h">—</b></div>
      <div class="mkt-stat"><span>Low</span><b id="mkt-l">—</b></div>
      <div class="mkt-stat"><span>Close (live)</span><b id="mkt-c">—</b></div>
      <div class="mkt-stat"><span>Candle closes in</span><b id="mkt-countdown">—</b></div>
      <div class="mkt-stat"><span>Updated</span><b id="mkt-updated">—</b></div>
    </div>
  </div>
  <div class="mkt-note muted">Live market data is shown for visualization only.
  This dashboard does not execute trades.</div>
</section>"""


def _data_sig(all_df: pd.DataFrame) -> str:
    """Cheap signature that changes only when a prediction is added or resolved.
    Lets the frontend skip re-applying section HTML when nothing has changed."""
    if all_df.empty:
        return "0:0:0"
    n = len(all_df)
    resolved = int(all_df["actual_direction"].notna().sum())
    mx = int(all_df["id"].max())
    return f"{n}:{resolved}:{mx}"


# ---------------------------------------------------------------------- #
# Theme (light fintech) + interactions
# ---------------------------------------------------------------------- #

_CSS = """
:root{
  --bg:#eef2f9; --surface:#ffffff; --surface-2:#f6f8fd; --border:#e7ecf5;
  --text:#0c1730; --muted:#5c6b86; --muted-2:#93a0b8;
  --accent:#3457d5; --accent-2:#5b7ff8; --accent-soft:#eaf0fe;
  --good:#16a34a; --good-soft:#e7f7ed;
  --bad:#dc2626; --bad-soft:#fdecec;
  --warn:#dd8000; --warn-soft:#fdf3e3;
  --info:#0e8fb3; --info-soft:#e4f4f9;
  --neutral:#5c6b86;
  --radius:16px; --radius-sm:11px;
  --shadow-sm:0 1px 2px rgba(16,24,40,.05);
  --shadow:0 1px 2px rgba(16,24,40,.04), 0 6px 18px rgba(16,24,40,.06);
  --shadow-lg:0 16px 40px rgba(16,24,40,.12);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);
  font:400 15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
h1,h2,h3{margin:0;font-weight:680;letter-spacing:-.01em;color:var(--text)}
h2{font-size:17px} h3{font-size:14px}
.muted{color:var(--muted);font-size:12.5px}
.nowrap{white-space:nowrap}
a{color:var(--accent);text-decoration:none}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}

.shell{display:flex;min-height:100vh}
.rail{position:sticky;top:0;height:100vh;width:64px;flex:0 0 64px;background:var(--surface);
  border-right:1px solid var(--border);display:flex;flex-direction:column;align-items:center;
  gap:6px;padding:16px 0}
.rail-logo{width:38px;height:38px;border-radius:11px;display:grid;place-items:center;
  color:#fff;font-size:18px;background:linear-gradient(135deg,#2563eb,#0e8fb3);margin-bottom:8px}
.rail-item{width:40px;height:40px;border-radius:11px;display:grid;place-items:center;
  color:var(--muted);transition:.15s}
.rail-item:hover{background:var(--accent-soft);color:var(--accent)}
.content{flex:1;min-width:0;max-width:1200px;margin:0 auto;padding:22px 26px 70px;width:100%}
.content>*{margin-top:18px}

.topbar{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-start;
  margin-top:4px}
.topbar h1{font-size:21px}
.subtitle{margin:3px 0 0;color:var(--muted);font-size:13px;max-width:56ch}
.chips{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.chip{font-size:11.5px;color:var(--muted);background:var(--surface);border:1px solid var(--border);
  border-radius:999px;padding:6px 12px;box-shadow:var(--shadow-sm);white-space:nowrap}
.chip-muted{color:var(--muted-2)}
.chip-live{color:#0a7d3f;border-color:#bfe6cd;background:var(--good-soft)}
.chip-live .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--good);
  margin-right:6px;vertical-align:middle;animation:pulse 1.6s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 20px;box-shadow:var(--shadow)}
.card-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:4px}
.card-title{font-weight:680}
.card-note{margin:2px 0 14px}
.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}

.notice{display:flex;gap:14px;background:var(--info-soft);border:1px solid #cfeaf3;
  border-radius:var(--radius);padding:14px 18px}
.notice-error{background:var(--bad-soft);border-color:#f6cccc}
.notice-icon{font-size:20px}
.notice-title{font-weight:680;margin-bottom:2px}
.notice-body{color:var(--muted);font-size:13px}

.verdict{display:flex;gap:16px;align-items:center;border-radius:var(--radius);padding:16px 20px;
  border:1px solid var(--border);box-shadow:var(--shadow);background:var(--surface)}
.verdict[data-tone=good]{background:var(--good-soft);border-color:#bfe6cd}
.verdict[data-tone=warn]{background:var(--warn-soft);border-color:#f6dcae}
.verdict[data-tone=bad]{background:var(--bad-soft);border-color:#f6cccc}
.verdict[data-tone=info]{background:var(--info-soft);border-color:#cfeaf3}
.verdict-icon{font-size:26px}
.verdict-head{font-size:18px;font-weight:700}
.verdict-sub{color:var(--muted);font-size:13px;margin-top:2px;max-width:82ch}

.badge{display:inline-block;font-size:11px;font-weight:650;padding:2px 9px;border-radius:999px;
  border:1px solid transparent;white-space:nowrap;vertical-align:middle}
.badge-good{color:#0a7d3f;background:var(--good-soft);border-color:#bfe6cd}
.badge-warn{color:#a15b00;background:var(--warn-soft);border-color:#f6dcae}
.badge-bad{color:#b21c1c;background:var(--bad-soft);border-color:#f6cccc}
.badge-info{color:#0a6f8c;background:var(--info-soft);border-color:#cfeaf3}
.badge-neutral,.badge-plain{color:var(--muted);background:var(--surface-2);border-color:var(--border)}

.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:14px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:16px 18px;box-shadow:var(--shadow)}
.kpi-top{display:flex;align-items:center;gap:8px}
.kpi-ico{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;
  color:var(--accent);background:var(--accent-soft)}
.kpi-ico[data-tone=good]{color:var(--good);background:var(--good-soft)}
.kpi-ico[data-tone=warn]{color:var(--warn);background:var(--warn-soft)}
.kpi-ico[data-tone=bad]{color:var(--bad);background:var(--bad-soft)}
.kpi-label{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.kpi-badge{margin-left:auto}
.kpi-value{font-size:29px;font-weight:720;margin:10px 0 2px;letter-spacing:-.02em}
.kpi[data-tone=good] .kpi-value{color:var(--good)}
.kpi[data-tone=warn] .kpi-value{color:var(--warn)}
.kpi[data-tone=bad] .kpi-value{color:var(--bad)}
.kpi-help{font-size:11.5px;color:var(--muted-2)}

.bar{display:block;height:8px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:999px;overflow:hidden;width:100%}
.bar-fill{display:block;height:100%;border-radius:999px}

.hstats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px}
.hstat{background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:12px}
.hstat-v{font-size:20px;font-weight:700}
.hstat-l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}

.rank-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.rank-card{background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:13px 14px}
.rank-l{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.rank-v{font-weight:680;font-size:16px;margin-top:4px}
.rank-a{font-weight:720;font-size:18px}
.rank-empty{margin-top:8px}

/* coin accordion */
.coin-list{display:flex;flex-direction:column;gap:12px;margin-top:12px}
.coin{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);overflow:hidden}
.coin>summary{list-style:none;cursor:pointer;display:grid;
  grid-template-columns:auto 1fr auto auto 140px auto auto;gap:14px;align-items:center;
  padding:16px 18px}
.coin>summary::-webkit-details-marker{display:none}
.coin-ava{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;color:#fff;
  font-weight:700;font-size:12px;letter-spacing:.02em}
.coin-id{display:flex;flex-direction:column;line-height:1.25}
.coin-sym{font-weight:720;font-size:16px}
.coin-acc{font-weight:720;font-size:18px;font-variant-numeric:tabular-nums}
.coin-bar{width:130px}
.coin-counts{white-space:nowrap}
.chev{color:var(--muted);transition:transform .2s;font-size:18px}
.coin[open] .chev{transform:rotate(180deg)}
.coin-body{padding:4px 18px 20px;border-top:1px solid var(--border)}
.coin-kv{display:flex;flex-wrap:wrap;gap:8px 22px;padding:14px 0;color:var(--muted);font-size:13px}
.coin-kv b{color:var(--text)}
.sub-h{margin:10px 0 10px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;font-size:11.5px}

.tf-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tf-card{background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:13px 14px}
.tf-top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.tf-name{font-weight:700}
.tf-acc{font-size:22px;font-weight:720;margin:6px 0 8px;font-variant-numeric:tabular-nums}
.tf-meta{display:flex;justify-content:space-between;font-size:12px;margin-top:9px}
.tf-sample{margin-top:8px}

/* filters */
.filters{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 14px}
.fgroup{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.flabel{font-size:11px;color:var(--muted-2);text-transform:uppercase;letter-spacing:.04em;margin-right:2px}
.fbtn{font-size:12px;color:var(--muted);background:var(--surface-2);border:1px solid var(--border);
  border-radius:999px;padding:5px 12px;cursor:pointer}
.fbtn:hover{color:var(--text)}
.fbtn.active{color:#fff;background:var(--accent);border-color:var(--accent);font-weight:600}

/* tables */
.table-scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
.log th,.log td{padding:10px 11px;border-bottom:1px solid var(--border);text-align:left}
.log thead th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted-2);font-weight:600}
.log tbody tr:hover{background:var(--surface-2)}
.confusion{font-size:12px;margin-top:6px}
.confusion td,.confusion th{border:1px solid var(--border);padding:5px 9px;text-align:center}
.confusion thead th{background:var(--surface-2);color:var(--muted)}
.pd-grid{display:flex;flex-wrap:wrap;gap:18px}
.pd-h{font-weight:650;font-size:12.5px;margin-bottom:2px}

/* controls */
.ctl-panel{display:flex;flex-wrap:wrap;gap:12px 16px;align-items:center;justify-content:space-between}
.ctl-status{display:flex;gap:8px;align-items:center}
.ctl-actions{display:flex;gap:8px;flex-wrap:wrap}
.ctl-form{display:inline}
.ctl-msg{font-size:12px;color:var(--muted);width:100%}
.btn{font-size:13px;font-weight:600;border:1px solid var(--border);border-radius:10px;
  padding:9px 15px;cursor:pointer;background:var(--surface);color:var(--text);box-shadow:var(--shadow-sm)}
.btn:hover{background:var(--surface-2)}
.btn-primary{background:var(--accent);color:#fff;border-color:transparent}
.btn-primary:hover{filter:brightness(1.05);background:var(--accent)}
.btn-ghost{background:var(--surface-2)}
.btn-warn{background:var(--warn-soft);color:#a15b00;border-color:#f6dcae}
.btn-danger{background:var(--bad-soft);color:#b21c1c;border-color:#f6cccc}

/* disclosures */
.disc>summary{list-style:none;cursor:pointer;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.disc>summary::-webkit-details-marker{display:none}
.disc>summary::before{content:"▸";color:var(--muted)}
.disc[open]>summary::before{content:"▾"}
.disc.sub{margin-top:14px;background:var(--surface-2);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:8px 14px}
.settings-form{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;
  margin-top:14px;align-items:end}
.settings-form label{display:flex;flex-direction:column;gap:5px;font-size:13px;font-weight:600}
.settings-form input{background:var(--surface);border:1px solid var(--border);border-radius:9px;
  padding:9px 11px;color:var(--text);font-size:13px}
.how-body{margin-top:12px}
.how-body ul{margin:0;padding-left:18px;color:var(--muted);font-size:13px;line-height:1.7}
.how-body b{color:var(--text)}

.empty{padding:26px;text-align:center;color:var(--muted);background:var(--surface-2);
  border:1px dashed var(--border);border-radius:var(--radius-sm)}

/* modal */
.modal{position:fixed;inset:0;background:rgba(15,27,52,.42);display:grid;place-items:center;z-index:50}
.modal[hidden]{display:none}
.modal-card{background:var(--surface);border-radius:var(--radius);padding:22px 24px;max-width:420px;
  width:calc(100% - 40px);box-shadow:0 20px 60px rgba(15,27,52,.3)}
.modal-title{font-weight:720;font-size:17px;margin-bottom:6px}
.modal-msg{color:var(--muted);font-size:13.5px;margin-bottom:18px}
.modal-actions{display:flex;justify-content:flex-end;gap:10px}

footer{margin-top:34px;color:var(--muted-2);font-size:12px;text-align:center}

@media(max-width:900px){
  .rail{display:none}
  .coin>summary{grid-template-columns:auto 1fr auto;row-gap:8px}
  .coin-bar,.coin-counts{grid-column:1 / -1;width:100%}
  .coin-bar{width:100%}
}
@media(max-width:560px){
  .hide-sm{display:none}
  .content{padding:18px 14px 54px}
  .kpi-value{font-size:25px}
}

/* ===== UI polish (presentation only — markup, classes & behaviour unchanged) ===== */
body{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  background:
    radial-gradient(900px 480px at 100% -6%, rgba(52,87,213,.06), transparent 60%),
    radial-gradient(680px 380px at -8% 2%, rgba(14,143,179,.05), transparent 55%),
    var(--bg);
  background-attachment:fixed}
h1{letter-spacing:-.02em}
.topbar h1{font-size:22px}
.rail-logo{box-shadow:0 6px 16px rgba(52,87,213,.35)}

.card,.kpi,.coin,.verdict,.chip,.btn,.fbtn,.tf-card,.rank-card,.rail-item,.coin>summary,.log tbody tr{
  transition:box-shadow .18s ease,transform .18s ease,background .18s ease,border-color .18s ease,filter .18s ease}
.card:hover,.kpi:hover,.rank-card:hover,.tf-card:hover{box-shadow:var(--shadow-lg);transform:translateY(-2px)}

/* KPI cards: hover lift + tone accent line + aligned figures */
.kpi{position:relative;overflow:hidden}
.kpi::after{content:"";position:absolute;left:0;right:0;top:0;height:3px;opacity:0;transition:opacity .18s;
  background:linear-gradient(90deg,var(--accent),var(--accent-2))}
.kpi:hover::after{opacity:.9}
.kpi[data-tone=good]::after{background:linear-gradient(90deg,var(--good),#39c46c)}
.kpi[data-tone=warn]::after{background:linear-gradient(90deg,var(--warn),#f5a524)}
.kpi[data-tone=bad]::after{background:linear-gradient(90deg,var(--bad),#f2557a)}
.kpi-value,.coin-acc,.tf-acc,.hstat-v,.rank-a,.log td{font-variant-numeric:tabular-nums}

/* buttons */
.btn{border-radius:11px}
.btn:hover{transform:translateY(-1px);box-shadow:var(--shadow)}
.btn:active{transform:translateY(0)}
.btn-primary{background:linear-gradient(180deg,var(--accent-2),var(--accent));border-color:transparent;
  box-shadow:0 4px 12px rgba(52,87,213,.28)}
.btn-primary:hover{filter:brightness(1.05);background:linear-gradient(180deg,var(--accent-2),var(--accent))}
.btn:focus-visible,.fbtn:focus-visible,.coin>summary:focus-visible,.settings-form input:focus-visible,.rail-item:focus-visible{
  outline:none;box-shadow:0 0 0 3px rgba(52,87,213,.22)}

/* coin accordion */
.coin>summary:hover{background:var(--surface-2)}
.coin[open]{box-shadow:var(--shadow-lg)}
.coin:hover{border-color:#d9e1f0}
.coin-ava{box-shadow:0 4px 10px rgba(16,24,40,.18)}

/* timeframe cards: tone accent bar */
.tf-card{position:relative;overflow:hidden}
.tf-card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--muted-2);opacity:.55}
.tf-card[data-tone=good]::before{background:var(--good);opacity:1}
.tf-card[data-tone=warn]::before{background:var(--warn);opacity:1}
.tf-card[data-tone=bad]::before{background:var(--bad);opacity:1}
.tf-card[data-tone=info]::before{background:var(--info);opacity:1}

/* verdict banner accent */
.verdict{position:relative;overflow:hidden;padding-left:22px}
.verdict::before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--info)}
.verdict[data-tone=good]::before{background:var(--good)}
.verdict[data-tone=warn]::before{background:var(--warn)}
.verdict[data-tone=bad]::before{background:var(--bad)}

/* tables, badges, filters, rail */
.log tbody tr:hover{background:var(--accent-soft)}
.badge{padding:3px 10px;font-weight:600}
.fbtn:hover{background:var(--accent-soft);border-color:#cdd9f6;color:var(--accent)}
.fbtn.active{box-shadow:0 3px 10px rgba(52,87,213,.26)}
.rail-item.active{background:var(--accent-soft);color:var(--accent)}

/* modal: soft blur + entrance */
.modal{backdrop-filter:blur(3px)}
.modal-card{box-shadow:var(--shadow-lg);animation:pop .16s ease-out}
@keyframes pop{from{transform:translateY(10px) scale(.98);opacity:0}to{transform:none;opacity:1}}

@media (prefers-reduced-motion: reduce){
  *{animation-duration:.001ms!important;transition:none!important}
  .card:hover,.kpi:hover,.rank-card:hover,.tf-card:hover,.btn:hover{transform:none}
}

/* ===== UI polish round 2 (presentation only — no markup/behaviour change) ===== */

/* roomier sections & consistent titles */
.content>*{margin-top:20px}
h2{font-size:17.5px}
.card{padding:20px 22px}

/* glassy white cards with a whisper of gradient */
.card,.kpi,.coin,.modal-card{background-image:linear-gradient(180deg,#ffffff,#fbfcff);border-color:#e9edf6}
.tf-card,.rank-card,.hstat{background-image:linear-gradient(180deg,#fbfcff,#f4f7fd)}

/* header refinements */
.topbar{align-items:center}
.subtitle{max-width:60ch;line-height:1.5}
.chip{padding:7px 13px;font-weight:500}
.chip-live{font-weight:600}
.chip-muted{background:var(--surface-2)}

/* controls: visually separate DANGEROUS actions from safe ones */
.ctl-actions{align-items:center}
.ctl-actions form:has(.btn-warn){margin-left:14px;padding-left:16px;border-left:1px solid var(--border)}
.btn-danger:hover{background:var(--bad);color:#fff;border-color:var(--bad);
  box-shadow:0 4px 12px rgba(220,38,38,.28)}
.btn-warn:hover{background:var(--warn);color:#fff;border-color:var(--warn)}

/* recent-predictions table: bounded scroll + sticky header (main log only) */
.table-scroll:has(#log-table){max-height:560px;overflow:auto;border:1px solid var(--border);
  border-radius:var(--radius-sm)}
.table-scroll:has(#log-table) .log thead th{position:sticky;top:0;z-index:2;
  background:#f5f8fd;border-bottom:1px solid var(--border)}
.log tbody tr:last-child td{border-bottom:none}

/* smooth reveal when a coin / disclosure opens */
details[open]>.coin-body,details.disc[open]>*:not(summary){animation:reveal .22s ease}
@keyframes reveal{from{opacity:0;transform:translateY(-5px)}to{opacity:1;transform:none}}

/* crisper badges + slightly taller bars */
.badge-good{box-shadow:inset 0 0 0 1px rgba(22,163,74,.12)}
.badge-bad{box-shadow:inset 0 0 0 1px rgba(220,38,38,.12)}
.badge-warn{box-shadow:inset 0 0 0 1px rgba(221,128,0,.14)}
.bar{height:9px}

/* mobile: full-width tappable buttons; dangerous group wraps below a divider */
@media(max-width:620px){
  .ctl-panel{align-items:stretch}
  .ctl-actions{width:100%}
  .ctl-actions form,.ctl-actions>.btn{flex:1 1 100%}
  .ctl-actions .btn{width:100%}
  .ctl-actions form:has(.btn-warn){margin-left:0;padding-left:0;border-left:none;
    margin-top:6px;padding-top:10px;border-top:1px solid var(--border)}
}

/* live-refresh status chip states (partial AJAX indicator) */
#live-chip{transition:background .2s,border-color .2s,color .2s}
#live-chip[data-state=busy]{color:#a15b00;background:var(--warn-soft);border-color:#f6dcae}
#live-chip[data-state=busy] .dot{background:var(--warn)}
#live-chip[data-state=err]{color:#b21c1c;background:var(--bad-soft);border-color:#f6cccc}
#live-chip[data-state=err] .dot{background:var(--bad);animation:none}
.chip-btn{cursor:pointer;font-family:inherit}
.chip-btn:hover{background:var(--accent-soft);color:var(--accent);border-color:#cdd9f6}
.chip-sel{font-weight:700;color:var(--accent);background:var(--accent-soft);border-color:#cdd9f6}

/* ===== Next-candle prediction hero ===== */
.hero{padding:0;overflow:hidden}
.hero .hero-head-row{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;
  padding:18px 22px 8px;flex-wrap:wrap}
.hero-kicker{font-size:11px;letter-spacing:.11em;color:var(--muted-2);font-weight:700;margin-bottom:9px}
.hero-sel{display:flex;gap:10px;flex-wrap:wrap}
.seg{display:inline-flex;background:var(--surface-2);border:1px solid var(--border);
  border-radius:12px;padding:3px;gap:2px}
.segbtn{border:none;background:transparent;color:var(--muted);font-weight:650;font-size:13px;
  padding:6px 14px;border-radius:9px;cursor:pointer;transition:.15s;font-family:inherit}
.segbtn:hover{color:var(--text)}
.segbtn.active{background:var(--surface);color:var(--accent);box-shadow:var(--shadow-sm)}
.hero-when{font-size:12px;padding-top:4px;white-space:nowrap}
.hero-grid{display:grid;grid-template-columns:1.15fr 1fr 1.5fr;align-items:stretch}
.hero-call{display:flex;gap:16px;align-items:center;padding:12px 22px 22px}
.hero-arrow{width:66px;height:66px;flex:0 0 66px;border-radius:18px;display:grid;place-items:center;
  font-size:32px;font-weight:800;color:#fff;background:#94a3b8;box-shadow:var(--shadow)}
.hero-call[data-dir=up] .hero-arrow{background:linear-gradient(180deg,#22c55e,#15a344)}
.hero-call[data-dir=down] .hero-arrow{background:linear-gradient(180deg,#f2557a,#dc2626)}
.hero-verdict{font-size:20px;font-weight:760;letter-spacing:-.01em;line-height:1.2}
.hero-call[data-dir=up] .hero-verdict{color:var(--good)}
.hero-call[data-dir=down] .hero-verdict{color:var(--bad)}
.hero-meta{font-size:12.5px;margin:3px 0 8px}
.hero-facts{display:grid;grid-template-columns:1fr 1fr;gap:14px 18px;padding:16px 20px;
  border-left:1px solid var(--border);border-right:1px solid var(--border);align-content:center}
.fact-l{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted-2);margin-bottom:2px}
.fact-v{font-weight:650;font-size:14px;font-variant-numeric:tabular-nums}
.pbar{display:flex;height:8px;width:130px;border-radius:999px;overflow:hidden;background:#eef2f7;margin-top:4px}
.pbar i{display:block;height:100%}
.pkey{font-size:10px;margin-top:4px}
.hero-chartwrap{padding:14px 20px 18px;position:relative;display:flex;flex-direction:column;min-width:0}
.hero-chart-top{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;color:var(--muted);
  margin-bottom:6px}
.hero-svg{width:100%;height:150px;display:block}
.hero-status .badge{margin-top:2px}

/* the coin/timeframe are chosen in the hero, so hide the log's duplicate filters */
.fgroup[data-group="coin"],.fgroup[data-group="tf"]{display:none}

/* ===== live candlestick chart ===== */
.mkt{padding:18px 22px}
.mkt-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.mkt-title{font-size:16px;font-weight:750;display:flex;gap:10px;align-items:center}
.mkt-state{font-size:11px;color:var(--muted);font-weight:500}
.mkt-price{text-align:right;white-space:nowrap}
.mkt-price-v{font-size:22px;font-weight:760;font-variant-numeric:tabular-nums}
.mkt-dir{margin-left:8px;vertical-align:middle}
.mkt-grid{display:grid;grid-template-columns:1fr 210px;gap:18px;margin-top:12px;align-items:stretch}
.mkt-chartwrap{position:relative;min-width:0}
.mkt-svg{width:100%;height:300px;display:block;
  background:linear-gradient(180deg,#fbfcff,#f5f8fd);border:1px solid var(--border);border-radius:var(--radius-sm)}
.mkt-signal{position:absolute;top:12px;left:12px;display:flex;gap:7px;align-items:center;
  background:rgba(255,255,255,.92);border:1px solid var(--border);border-radius:999px;
  padding:5px 13px;font-size:12.5px;font-weight:650;box-shadow:var(--shadow-sm)}
.mkt-signal .ar{font-size:14px}
.mkt-signal[data-dir=up]{color:var(--good);border-color:#bfe6cd}
.mkt-signal[data-dir=down]{color:var(--bad);border-color:#f6cccc}
.mkt-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  background:var(--surface-2);border-radius:var(--radius-sm)}
.mkt-empty[hidden]{display:none}
.mkt-panel{display:flex;flex-direction:column;gap:8px;justify-content:center}
.mkt-stat{display:flex;justify-content:space-between;gap:10px;font-size:13px;padding:9px 13px;
  background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm)}
.mkt-stat span{color:var(--muted)}
.mkt-stat b{font-variant-numeric:tabular-nums}
.mkt-note{margin-top:12px;font-size:11.5px}
@media(max-width:820px){ .mkt-grid{grid-template-columns:1fr} .mkt-svg{height:240px} }

@media(max-width:900px){
  .hero-grid{grid-template-columns:1fr}
  .hero-facts{border-left:none;border-right:none;
    border-top:1px solid var(--border);border-bottom:1px solid var(--border)}
}
"""

_JS = """
<script>
(function(){
  "use strict";
  var REFRESH=parseInt((document.body.getAttribute('data-refresh')||'0'),10);
  // Only these data sections are swapped on live refresh. The header, research
  // notice, controls panel, settings panel and modal are NEVER touched, so
  // typing, the predictor status and open modals are preserved.
  var SECTIONS=['sec-verdict','overview','sec-validation','sec-rankings','coins','log'];
  var filterState={coin:'all',tf:'all',result:'all'};

  function applyFilters(){
    document.querySelectorAll('.fbtn').forEach(function(x){
      var f=x.getAttribute('data-f');
      x.classList.toggle('active', x.getAttribute('data-v')===filterState[f]);
    });
    var rows=document.querySelectorAll('#log-table [data-row]'), shown=0;
    rows.forEach(function(r){
      var ok=(filterState.coin==='all'||r.getAttribute('data-coin')===filterState.coin)
        &&(filterState.tf==='all'||r.getAttribute('data-tf')===filterState.tf)
        &&(filterState.result==='all'||r.getAttribute('data-result')===filterState.result);
      r.style.display=ok?'':'none'; if(ok)shown++;
    });
    var e=document.querySelector('.filter-empty'); if(e)e.hidden=shown!==0;
  }
  function wireAccordions(){
    var coins=[].slice.call(document.querySelectorAll('details.coin'));
    coins.forEach(function(d){
      if(d.__wired)return; d.__wired=true;
      d.addEventListener('toggle',function(){
        if(d.open){coins.forEach(function(o){if(o!==d)o.open=false;});}
      });
    });
  }
  function wireFilters(){
    document.querySelectorAll('.fbtn').forEach(function(b){
      if(b.__wired)return; b.__wired=true;
      b.addEventListener('click',function(){
        filterState[b.getAttribute('data-f')]=b.getAttribute('data-v');
        applyFilters();
      });
    });
  }
  function wireModal(){
    var modal=document.getElementById('modal');
    if(!modal||modal.__wired)return; modal.__wired=true;
    var msg=document.getElementById('modal-msg'), pending=null;
    document.querySelectorAll('form[data-confirm]').forEach(function(f){
      f.addEventListener('submit',function(e){
        if(f.dataset.ok==='1')return;
        e.preventDefault(); pending=f;
        if(msg)msg.textContent=f.getAttribute('data-confirm'); modal.hidden=false;
      });
    });
    var ok=document.getElementById('modal-ok'), cancel=document.getElementById('modal-cancel');
    if(ok)ok.onclick=function(){if(pending){pending.dataset.ok='1';pending.submit();}};
    if(cancel)cancel.onclick=function(){modal.hidden=true;pending=null;};
    modal.addEventListener('click',function(e){if(e.target===modal){modal.hidden=true;pending=null;}});
  }
  function wireAll(){wireAccordions();wireFilters();wireModal();}

  /* ---------- next-candle prediction hero + chart (existing data only) ---------- */
  function readData(){
    try{return JSON.parse(document.getElementById('dash-data').textContent);}
    catch(e){return {coins:[],tfs:{},pred:{}};}
  }
  var DATA=readData();
  var selCoin='', selTf='';
  try{selCoin=localStorage.getItem('tbv_coin')||'';selTf=localStorage.getItem('tbv_tf')||'';}catch(e){}
  function ensureSel(){
    if(!DATA.coins.length){selCoin='';selTf='';return;}
    if(DATA.coins.indexOf(selCoin)<0)selCoin=DATA.coins[0];
    var tfs=DATA.tfs[selCoin]||[];
    if(tfs.indexOf(selTf)<0)selTf=tfs[0]||'';
  }
  function saveSel(){try{localStorage.setItem('tbv_coin',selCoin);localStorage.setItem('tbv_tf',selTf);}catch(e){}}
  function fmtPrice(v){
    if(v==null)return '—'; var a=Math.abs(v); var d=a>=100?2:a>=1?4:6;
    return Number(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
  }
  function seg(id,items,cur,attr,on){
    var el=document.getElementById(id); if(!el)return;
    el.innerHTML=items.map(function(v){
      return '<button class="segbtn'+(v===cur?' active':'')+'" type="button" data-'+attr+'="'+v+'">'+v+'</button>';
    }).join('');
    el.querySelectorAll('.segbtn').forEach(function(b){
      b.addEventListener('click',function(){on(b.getAttribute('data-'+attr));});
    });
  }
  function buildSegs(){
    seg('coin-seg',DATA.coins,selCoin,'coin',function(c){selCoin=c;ensureSel();saveSel();renderAll();});
    seg('tf-seg',(DATA.tfs[selCoin]||[]),selTf,'tf',function(t){selTf=t;saveSel();renderAll();});
  }
  // --- granular value updaters (never rebuild structure -> no flicker) ---
  function setTxt(id,t){var e=document.getElementById(id);if(e&&e.textContent!==t)e.textContent=t;}
  function setW(id,w,bg){var e=document.getElementById(id);if(e){var s=w+'%';if(e.style.width!==s)e.style.width=s;if(bg&&e.style.background!==bg)e.style.background=bg;}}
  function setAttr(el,a,v){if(el&&el.getAttribute(a)!==v)el.setAttribute(a,v);}
  function updateHeroValues(){
    var sc=document.getElementById('sel-chip');
    if(sc){ if(selCoin){sc.hidden=false;setTxt('sel-chip',selCoin+' · '+selTf);} else sc.hidden=true; }
    var call=document.getElementById('hero-call');
    var p=(DATA.pred[selCoin]||{})[selTf];
    if(!p){
      setAttr(call,'data-dir','flat'); setTxt('hero-arrow','–'); setTxt('hero-verdict','No prediction yet');
      setTxt('hero-meta',selCoin?(selCoin+' · '+selTf):''); setTxt('hero-target','—');
      var s0=document.getElementById('hero-status'); if(s0&&s0.innerHTML!=='')s0.innerHTML='';
      setTxt('f-conf','—'); setTxt('f-exp','—'); setTxt('f-last','—'); setTxt('pb-key','—');
      setW('pb-up',0); setW('pb-neu',0); setW('pb-down',0); return;
    }
    setAttr(call,'data-dir',p.up?'up':p.down?'down':'flat');
    setTxt('hero-arrow',p.up?'▲':p.down?'▼':'■');
    setTxt('hero-verdict',p.up?'Next candle predicted UP':p.down?'Next candle predicted DOWN':'Next candle predicted FLAT');
    setTxt('hero-meta',selCoin+' · '+selTf+' · '+Math.round(p.conf*100)+'% '+(p.conf_label||''));
    setTxt('hero-target',p.target||'—');
    var st=document.getElementById('hero-status');
    var sh = p.status==='pending' ? '<span class="badge badge-warn">Pending · waiting for candle</span>'
      : (p.correct ? '<span class="badge badge-good">Resolved · correct</span>'
                   : '<span class="badge badge-bad">Resolved · incorrect</span>');
    if(st&&st.innerHTML!==sh)st.innerHTML=sh;
    setTxt('f-conf',Math.round(p.conf*100)+'%'+(p.conf_label?(' '+p.conf_label):''));
    setW('pb-up',(p.bull*100),'var(--good)'); setW('pb-neu',(p.neu*100),'#94a3b8'); setW('pb-down',(p.bear*100),'var(--bad)');
    setTxt('pb-key','up '+Math.round(p.bull*100)+'% · flat '+Math.round(p.neu*100)+'% · down '+Math.round(p.bear*100)+'%');
    setTxt('f-exp',(p.exp_low!=null)?(fmtPrice(p.exp_low)+' – '+fmtPrice(p.exp_high)):'—');
    setTxt('f-last',fmtPrice(p.ref));
  }
  function updateChart(){
    var p=(DATA.pred[selCoin]||{})[selTf]; var s=(p&&p.series)?p.series:[];
    setTxt('chart-label',selCoin?(selCoin+' '+selTf+' · recent close'):'Recent price');
    var empty=document.getElementById('chart-empty');
    var area=document.getElementById('ch-area'),line=document.getElementById('ch-line'),dot=document.getElementById('ch-dot');
    if(!line)return;
    if(s.length<2){ if(empty)empty.hidden=false; setAttr(area,'d',''); setAttr(line,'d',''); setAttr(dot,'cx','-10'); setTxt('chart-range',''); return; }
    if(empty)empty.hidden=true;
    var W=600,H=150,pad=10;
    var min=Math.min.apply(null,s),max=Math.max.apply(null,s),span=(max-min)||1;
    var pts=s.map(function(v,i){return [pad+(W-2*pad)*(i/(s.length-1)), pad+(H-2*pad)*(1-(v-min)/span)];});
    var ld=pts.map(function(pt,i){return (i?'L':'M')+pt[0].toFixed(1)+' '+pt[1].toFixed(1);}).join(' ');
    var ad='M '+pts[0][0].toFixed(1)+' '+(H-pad);
    pts.forEach(function(pt){ad+=' L '+pt[0].toFixed(1)+' '+pt[1].toFixed(1);});
    ad+=' L '+pts[pts.length-1][0].toFixed(1)+' '+(H-pad)+' Z';
    var col=p.up?'#16a34a':p.down?'#dc2626':'#64748b', last=pts[pts.length-1];
    setAttr(area,'d',ad); setAttr(line,'d',ld); setAttr(line,'stroke',col);
    setAttr(dot,'cx',last[0].toFixed(1)); setAttr(dot,'cy',last[1].toFixed(1)); setAttr(dot,'fill',col);
    setAttr(document.getElementById('cg0'),'stop-color',col); setAttr(document.getElementById('cg1'),'stop-color',col);
    setTxt('chart-range',fmtPrice(min)+' – '+fmtPrice(max));
  }
  function applyCoinVisibility(){
    var any=false;
    document.querySelectorAll('details.coin').forEach(function(d){
      var s=d.querySelector('.coin-sym'); var sym=s?s.textContent.trim():'';
      var show=(sym===selCoin); if(show)any=true;
      d.style.display=show?'':'none'; d.open=show;
    });
    return any;
  }
  function syncRecent(){
    filterState.coin=selCoin||'all'; filterState.tf=selTf||'all'; applyFilters();
  }
  function renderAll(){ ensureSel(); buildSegs(); updateHeroValues(); updateChart(); applyCoinVisibility(); syncRecent(); fetchMarket(); }

  /* ---------- live candlestick chart (read-only Binance klines) ---------- */
  var mktBusy=false, mktCloseMs=0, mktHasData=false, mktKey='';
  function mktSym(){ return (DATA.sym&&DATA.sym[selCoin])||(selCoin?selCoin+'USDT':''); }
  function fmtDur(ms){ if(ms<0)ms=0; var s=Math.floor(ms/1000),m=Math.floor(s/60); s=s%60;
    return (m<10?'0':'')+m+':'+(s<10?'0':'')+s; }
  function mktErr(){
    setTxt('mkt-state','Live market data unavailable. Retrying…');
    if(!mktHasData){var e=document.getElementById('mkt-empty');
      if(e){e.hidden=false;e.textContent='Live market data unavailable. Retrying…';}}
  }
  function fetchMarket(){
    var sym=mktSym(), tf=selTf;
    if(!sym||!tf) return;
    if(mktBusy) return; mktBusy=true;
    // if the coin/tf changed since last draw, reset the "has data" guard so the
    // empty state can show while the new symbol loads
    var key=sym+'|'+tf; if(key!==mktKey){ mktKey=key; mktHasData=false;
      setTxt('mkt-pair',sym+' · '+tf); }
    fetch('/api/market-candles?symbol='+encodeURIComponent(sym)+'&timeframe='+encodeURIComponent(tf),
          {credentials:'same-origin',cache:'no-store'})
      .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
      .then(function(m){
        if(m.error || !m.candles || m.candles.length<1){ mktErr(); return; }
        if((m.symbol+'|'+m.timeframe)!==mktKey) return;   // selection changed mid-flight
        drawMarket(m);
      })
      .catch(function(){ mktErr(); })
      .then(function(){ mktBusy=false; });
  }
  function drawMarket(m){
    mktHasData=true;
    var e=document.getElementById('mkt-empty'); if(e)e.hidden=true;
    setTxt('mkt-state','live'); setTxt('mkt-pair',m.symbol+' · '+m.timeframe);
    var cs=m.candles, cur=cs[cs.length-1];
    mktCloseMs=cur.close_time+1;
    var price=(m.current_price!=null)?m.current_price:cur.close;
    setTxt('mkt-price',fmtPrice(price));
    setTxt('mkt-o',fmtPrice(cur.open)); setTxt('mkt-h',fmtPrice(cur.high));
    setTxt('mkt-l',fmtPrice(cur.low)); setTxt('mkt-c',fmtPrice(cur.close));
    setTxt('mkt-updated',m.updated_at||'—');
    var up=cur.close>=cur.open, dd=document.getElementById('mkt-dir');
    if(dd){ dd.textContent=up?'▲ up':'▼ down';
      dd.className='mkt-dir badge '+(up?'badge-good':'badge-bad'); }
    drawCandles(cs);
    var p=(DATA.pred[selCoin]||{})[selTf], sig=document.getElementById('mkt-signal');
    if(sig){
      if(p){ sig.hidden=false; sig.setAttribute('data-dir',p.up?'up':p.down?'down':'flat');
        sig.innerHTML='<span class="ar">'+(p.up?'▲':p.down?'▼':'■')+'</span> Model signal: '
          +(p.up?'UP':p.down?'DOWN':'FLAT')+' · '+Math.round(p.conf*100)+'%';
      } else sig.hidden=true;
    }
    updateCountdown();
  }
  function drawCandles(cs){
    var plot=document.getElementById('mkt-plot'), grid=document.getElementById('mkt-grid');
    if(!plot) return;
    var W=720,H=300,padT=10,padB=18,padL=6,padR=58;
    var lo=Infinity,hi=-Infinity;
    cs.forEach(function(c){ if(c.low<lo)lo=c.low; if(c.high>hi)hi=c.high; });
    if(!isFinite(lo)||!isFinite(hi)){ plot.innerHTML=''; return; }
    var pd=(hi-lo)*0.08||1; lo-=pd; hi+=pd; var span=(hi-lo)||1;
    function Y(v){ return padT+(H-padT-padB)*(1-(v-lo)/span); }
    var n=cs.length, cw=(W-padL-padR)/n;
    var g='';
    for(var k=0;k<=4;k++){ var yy=padT+(H-padT-padB)*k/4, pv=hi-(hi-lo)*k/4;
      g+='<line x1="'+padL+'" x2="'+(W-padR)+'" y1="'+yy.toFixed(1)+'" y2="'+yy.toFixed(1)+'" stroke="#e7ecf5"/>'
        +'<text x="'+(W-padR+5)+'" y="'+(yy+3).toFixed(1)+'" font-size="10" fill="#93a0b8">'+fmtPrice(pv)+'</text>';
    }
    grid.innerHTML=g;
    var b='';
    for(var i=0;i<n;i++){ var c=cs[i], xc=padL+i*cw+cw/2, up=c.close>=c.open,
        col=up?'#16a34a':'#dc2626', bw=Math.max(1,cw*0.62),
        yo=Y(c.open), ycl=Y(c.close), top=Math.min(yo,ycl), hgt=Math.max(1,Math.abs(yo-ycl));
      b+='<line x1="'+xc.toFixed(1)+'" x2="'+xc.toFixed(1)+'" y1="'+Y(c.high).toFixed(1)+'" y2="'+Y(c.low).toFixed(1)+'" stroke="'+col+'" stroke-width="1"/>'
        +'<rect x="'+(xc-bw/2).toFixed(1)+'" y="'+top.toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+hgt.toFixed(1)+'" rx="0.6" fill="'+col+'"'
        +(i===n-1?' stroke="#0c1730" stroke-opacity="0.18"':'')+'/>';
    }
    var cp=cs[n-1].close, cy=Y(cp);
    b+='<line x1="'+padL+'" x2="'+(W-padR)+'" y1="'+cy.toFixed(1)+'" y2="'+cy.toFixed(1)+'" stroke="#3457d5" stroke-width="1" stroke-dasharray="4 3" opacity="0.65"/>';
    plot.innerHTML=b;
  }
  function updateCountdown(){
    var el=document.getElementById('mkt-countdown'); if(!el)return;
    el.textContent = mktCloseMs ? fmtDur(mktCloseMs-Date.now()) : '—';
  }

  // "Settings" chip -> open + scroll to the settings panel (kept at the bottom)
  (function(){
    var jump=document.getElementById('settings-jump'), panel=document.getElementById('settings-panel');
    if(jump){ if(!panel){jump.hidden=true;} else { jump.hidden=false;
      jump.addEventListener('click',function(){panel.open=true;panel.scrollIntoView({behavior:'smooth',block:'start'});}); } }
  })();

  var statusEl=document.getElementById('refresh-status');
  var liveChip=document.getElementById('live-chip');
  // Silent by default: text stays "Live" (no "Refreshing…" flip). Only an
  // error changes it. This removes the header flicker / layout shift.
  function setStatus(kind){
    if(liveChip)liveChip.setAttribute('data-state',kind==='err'?'err':'ok');
    if(statusEl){var t=(kind==='err')?'Update failed, retrying…':'Live';
      if(statusEl.textContent!==t)statusEl.textContent=t;}
  }
  function logBox(){var t=document.getElementById('log-table');return t?t.closest('.table-scroll'):null;}

  var lastSig=(document.body.getAttribute('data-sig')||'');
  var busy=false;
  // Partial data refresh: hits the JSON endpoint, updates VALUES silently, and
  // re-applies section HTML ONLY when the data signature actually changed.
  function fetchData(){
    if(busy)return;
    var modal=document.getElementById('modal');
    if(modal && !modal.hidden) return;   // never disrupt an open confirmation
    busy=true;
    var scrollY=window.scrollY;
    var lb=logBox(); var logTop=lb?lb.scrollTop:0;
    fetch('/api/dashboard-data?sig='+encodeURIComponent(lastSig),
          {credentials:'same-origin',cache:'no-store'})
      .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
      .then(function(j){
        console.log("Partial dashboard data refresh");
        if(j.hero) DATA=j.hero;
        updateHeroValues(); updateChart();           // silent value-only updates
        if(j.updated) setTxt('last-updated','Updated '+j.updated);
        if(j.sig && j.sig!==lastSig && j.html){       // data changed -> swap sections
          lastSig=j.sig;
          Object.keys(j.html).forEach(function(id){
            var el=document.getElementById(id); if(el&&j.html[id])el.outerHTML=j.html[id];
          });
          wireAll(); applyCoinVisibility(); syncRecent();
          var lb2=logBox(); if(lb2) lb2.scrollTop=logTop;
          window.scrollTo(0,scrollY);
        } else if(j.sig){ lastSig=j.sig; }
        setStatus('ok');
      })
      .catch(function(){ setStatus('err'); })
      .then(function(){ busy=false; });
  }

  console.log("Initial dashboard load");
  wireAll(); renderAll(); setStatus('ok');
  if(REFRESH>0){ setInterval(fetchData, REFRESH*1000); }
  // live market chart: refresh candles ~every 2.5s, tick the countdown every 1s
  setInterval(fetchMarket, 2500);
  setInterval(updateCountdown, 1000);

  // Manual Refresh button -> same silent partial refresh (never a full reload)
  var rbtn=document.querySelector('a.btn[href="/"]');
  if(rbtn) rbtn.addEventListener('click',function(e){ e.preventDefault(); fetchData(); });
})();
</script>
"""


def build_dashboard_html(
    storage: PredictionStorage,
    refresh_seconds: int | None = None,
    offset_hours: float = 0.0,
    controls_enabled: bool = False,
    control_message: str = "",
    config_path: str | None = None,
) -> str:
    """Render the full dashboard page. Presentation only — data comes straight
    from the prediction log via :class:`PredictionStorage` and the evaluator."""
    try:
        all_df = storage.load_dataframe()
    except Exception as exc:  # data source unreadable -> styled error state
        return _error_page(f"{exc}", refresh_seconds)

    resolved = (
        all_df[all_df["actual_direction"].notna()].copy() if len(all_df) else all_df
    )
    report = (
        build_report(resolved)
        if len(resolved)
        else {
            "groups": [], "overall_accuracy_pct": 0.0, "total_resolved": 0,
            "best_timeframe": None, "worst_timeframe": None,
            "best_symbol": None, "worst_symbol": None,
        }
    )
    coins = _coin_groups(all_df, report)
    data_json = json.dumps(_dashboard_data(all_df, offset_hours)).replace("</", "<\\/")

    tz = display_tz_label(offset_hours)
    generated = to_display_time(datetime.now(timezone.utc), offset_hours)
    # Live updates are done in-browser via partial AJAX (see _JS) instead of a
    # full-page <meta refresh>, so open accordions / filters / scroll / typing
    # and open modals are preserved. This attribute carries the interval.
    refresh_attr = int(refresh_seconds) if refresh_seconds and refresh_seconds > 0 else 0

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Research dashboard for validating crypto next-candle prediction accuracy.">
<title>TradeBotVol | Prediction Dashboard</title>
<style>{_CSS}</style></head>
<body data-refresh="{refresh_attr}" data-sig="{_data_sig(all_df)}">
<div class="shell">
  {_sidebar()}
  <main class="content">
    {_header(generated, refresh_seconds, tz)}
    {_hero()}
    {_market_chart()}
    {_control_panel(controls_enabled, control_message)}
    {_kpis(all_df, resolved, report)}
    {_verdict(resolved, report)}
    {_validation_health(all_df, resolved, report)}
    {_rankings(report)}
    {_coin_accordion(coins, offset_hours)}
    {_recent_log(all_df, offset_hours)}
    {_settings_panel(controls_enabled, config_path)}
    {_how_to_read()}
    <footer>Prediction-only research tool · times in {_esc(tz)} ·
    judge from large samples, not small ones.</footer>
  </main>
</div>
<script type="application/json" id="dash-data">{data_json}</script>
{_modal()}
{_JS}
</body></html>"""


def write_dashboard(
    storage: PredictionStorage,
    out_path: str = "dashboard.html",
    refresh_seconds: int | None = None,
    offset_hours: float = 0.0,
) -> str:
    html_text = build_dashboard_html(storage, refresh_seconds, offset_hours)
    path = Path(out_path)
    path.write_text(html_text, encoding="utf-8")
    return str(path.resolve())


# ---------------------------------------------------------------------- #
# Live server (stdlib only — no Flask, no extra dependencies)
# ---------------------------------------------------------------------- #


def serve_dashboard(
    db_path: str,
    csv_path: str,
    host: str = "127.0.0.1",
    port: int = 8787,
    refresh_seconds: int = 5,
    offset_hours: float = 0.0,
    controls_enabled: bool = False,
    config_path: str | None = None,
    auth_user: str | None = None,
    auth_password_hash: str | None = None,
    protect_view: bool = True,
    market_type: str = "futures",
    allowed_symbols: list | None = None,
) -> None:
    """Serve a live dashboard that regenerates from the DB on every request.

    A separate process (``run_predictor.py``) writes new predictions to the
    same SQLite file; each browser refresh re-reads the latest committed rows.

    When ``controls_enabled`` is true, the page shows pause/resume/restart/clear/
    shutdown buttons and a settings form.

    Auth: when ``auth_user`` and ``auth_password_hash`` (sha256 hex) are both set,
    the state-changing POST routes always require HTTP Basic auth. ``GET`` (viewing
    the dashboard) requires auth only when ``protect_view`` is true; set it false
    for a public read-only dashboard whose control buttons still need a login.
    ``GET /health`` is always open. With no credentials set, the server is fully
    open (local development), preserving the original behaviour.
    """
    import base64
    import hashlib
    import hmac
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs

    from .logger import get_logger

    log = get_logger("dashboard")
    pending_message: dict[str, str] = {"text": ""}
    auth_required = bool(auth_user and auth_password_hash)
    start_monotonic = time.monotonic()
    allowed = set(allowed_symbols or [])
    # Read-only live market data (public Binance klines). Reuses the existing
    # BinanceDataClient; a short cache + lock keep Binance calls light even with
    # several browsers polling. No trading endpoints, no API keys.
    market = {"client": None, "cache": {}, "lock": __import__("threading").Lock()}

    def market_payload(symbol: str, timeframe: str) -> bytes:
        if allowed and symbol not in allowed:
            return json.dumps({"error": "unknown symbol", "symbol": symbol}).encode("utf-8")
        if timeframe not in TIMEFRAME_MINUTES:
            return json.dumps({"error": "unknown timeframe"}).encode("utf-8")
        key = (symbol, timeframe)
        now = time.time()
        with market["lock"]:
            cached = market["cache"].get(key)
            if cached and now - cached[0] < 1.5:
                return cached[1]
        try:
            if market["client"] is None:
                from .binance_data import BinanceDataClient
                market["client"] = BinanceDataClient(market_type)
            df = market["client"].get_klines(
                symbol, timeframe, limit=100, only_closed=False
            )
        except Exception as exc:
            return json.dumps({
                "error": "market data unavailable",
                "detail": str(exc)[:120], "symbol": symbol, "timeframe": timeframe,
            }).encode("utf-8")
        if df.empty:
            return json.dumps({
                "symbol": symbol, "timeframe": timeframe, "candles": [],
                "current_price": None,
            }).encode("utf-8")
        candles = [
            {"open_time": int(r.open_time_ms), "open": float(r.open),
             "high": float(r.high), "low": float(r.low), "close": float(r.close),
             "volume": float(r.volume), "close_time": int(r.close_time_ms)}
            for r in df.itertuples(index=False)
        ]
        payload = {
            "symbol": symbol, "timeframe": timeframe, "candles": candles,
            "current_price": float(df["close"].iloc[-1]),
            "updated_at": to_display_time(datetime.now(timezone.utc), offset_hours)
            + " " + display_tz_label(offset_hours),
            "server_time_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
        }
        body = json.dumps(payload).encode("utf-8")
        with market["lock"]:
            market["cache"][key] = (now, body)
        return body

    def authorized(headers) -> bool:
        if not auth_required:
            return True
        header = headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
        except Exception:
            return False
        pw_hash = hashlib.sha256(pw.encode("utf-8")).hexdigest()
        return hmac.compare_digest(user, auth_user or "") and hmac.compare_digest(
            pw_hash, auth_password_hash or ""
        )

    def health_payload() -> bytes:
        try:
            from . import __version__ as version
        except Exception:
            version = "unknown"
        return json.dumps(
            {
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "version": version,
                "uptime_seconds": round(time.monotonic() - start_monotonic),
            }
        ).encode("utf-8")

    def render() -> bytes:
        storage = PredictionStorage(
            db_path, csv_path, sqlite_enabled=True, csv_enabled=False
        )
        try:
            message = pending_message["text"]
            pending_message["text"] = ""
            return build_dashboard_html(
                storage, refresh_seconds, offset_hours, controls_enabled, message,
                config_path,
            ).encode("utf-8")
        finally:
            storage.close()

    def api_payload(client_sig: str) -> bytes:
        """JSON for the silent partial refresh. Reads the same prediction log as
        the page (no prediction/scoring/storage logic). Returns the hero data +
        a data signature every time, and the (rarely-changing) section HTML only
        when the signature differs from what the client already shows."""
        storage = PredictionStorage(
            db_path, csv_path, sqlite_enabled=True, csv_enabled=False
        )
        try:
            all_df = storage.load_dataframe()
        except Exception as exc:  # pragma: no cover - defensive
            storage.close()
            return json.dumps({"error": str(exc)}).encode("utf-8")
        try:
            resolved = (
                all_df[all_df["actual_direction"].notna()].copy()
                if len(all_df) else all_df
            )
            report = (
                build_report(resolved)
                if len(resolved)
                else {
                    "groups": [], "overall_accuracy_pct": 0.0, "total_resolved": 0,
                    "best_timeframe": None, "worst_timeframe": None,
                    "best_symbol": None, "worst_symbol": None,
                }
            )
            sig = _data_sig(all_df)
            tz = display_tz_label(offset_hours)
            updated = to_display_time(datetime.now(timezone.utc), offset_hours) + " " + tz
            payload = {
                "updated": updated,
                "sig": sig,
                "hero": _dashboard_data(all_df, offset_hours),
            }
            if sig != client_sig:  # data changed -> send fresh section HTML
                coins = _coin_groups(all_df, report)
                payload["html"] = {
                    "sec-verdict": _verdict(resolved, report),
                    "overview": _kpis(all_df, resolved, report),
                    "sec-validation": _validation_health(all_df, resolved, report),
                    "sec-rankings": _rankings(report),
                    "coins": _coin_accordion(coins, offset_hours),
                    "log": _recent_log(all_df, offset_hours),
                }
            return json.dumps(payload).encode("utf-8")
        finally:
            storage.close()

    class Handler(BaseHTTPRequestHandler):
        def _deny(self) -> None:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="TradeBotVol dashboard"')
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Authentication required")

        def do_POST(self) -> None:  # noqa: N802
            # Drain the request body first so the socket closes cleanly even on
            # a rejected request (avoids connection resets on some clients).
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            if not authorized(self.headers):  # protect state-changing routes
                self._deny()
                return
            if not controls_enabled or self.path not in ("/action", "/settings"):
                self.send_error(404)
                return
            raw = parse_qs(body.decode("utf-8"))
            if self.path == "/action":
                action = raw.get("action", [""])[0]
                log.info("Dashboard control action: %s", action)
                pending_message["text"] = run_control_action(action, db_path, csv_path)
            else:  # /settings
                form = {k: v[0] for k, v in raw.items()}
                log.info("Dashboard settings update")
                if config_path:
                    pending_message["text"] = apply_settings(config_path, form)
                else:
                    pending_message["text"] = "settings unavailable (no config path)"
            self.send_response(303)  # POST-redirect-GET
            self.send_header("Location", "/")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            # Health check is intentionally unauthenticated and exposes nothing
            # sensitive (no secrets, DB contents, logs or prediction internals).
            if self.path in ("/health", "/healthz"):
                body = health_payload()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            # Read-only JSON for the silent partial refresh (same access rule as
            # the page view). Exposes only prediction-log data — no secrets/logs.
            if self.path.split("?", 1)[0] == "/api/dashboard-data":
                if protect_view and not authorized(self.headers):
                    self._deny()
                    return
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                client_sig = parse_qs(query).get("sig", [""])[0]
                body = api_payload(client_sig)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            # Read-only live market candles (public Binance klines). No secrets,
            # no trading — validated against the configured symbols/timeframes.
            if self.path.split("?", 1)[0] == "/api/market-candles":
                if protect_view and not authorized(self.headers):
                    self._deny()
                    return
                q = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
                sym = (q.get("symbol", [""])[0] or "").upper()
                tfr = q.get("timeframe", [""])[0]
                body = market_payload(sym, tfr)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if protect_view and not authorized(self.headers):  # optional view gate
                self._deny()
                return
            if self.path not in ("/", "/index.html", "/dashboard.html"):
                self.send_error(404)
                return
            try:
                body = render()  # opens + closes its own DB connection
            except Exception as exc:  # pragma: no cover - defensive
                self.send_error(500, f"dashboard render failed: {exc}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass  # silence per-request stderr noise

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    log.info(
        "Live dashboard serving at %s (auto-refresh %ds, auth %s)",
        url,
        refresh_seconds,
        "on" if auth_required else "OFF",
    )
    print(f"Live dashboard: {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    finally:
        server.shutdown()
        server.server_close()
