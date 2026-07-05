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
            f'<span class="chip chip-live"><span class="dot"></span>'
            f"Live refresh: {int(refresh_seconds)}s</span>"
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
    <span class="chip">Source: SQLite / CSV prediction log</span>
    <span class="chip chip-muted">Updated {_esc(generated)} {_esc(tz)}</span>
  </div>
</header>"""


def _research_notice() -> str:
    return """
<div class="notice" role="note" aria-label="Research mode notice">
  <div class="notice-icon" aria-hidden="true">🛈</div>
  <div>
    <div class="notice-title">Research mode only</div>
    <div class="notice-body">This dashboard validates prediction accuracy only.
    It does not execute trades and should not be connected to live trading until
    accuracy is validated over a statistically significant sample.</div>
  </div>
</div>"""


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
<section class="verdict" data-tone="{tone}" aria-label="Current status">
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
<details class="card disc">
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
<section class="card" aria-label="Validation health">
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
<section class="card" aria-label="Model rankings">
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
  <div class="section-head"><h2>Performance by coin</h2>
    <span class="muted">tap a coin to expand its timeframes &amp; predictions</span></div>
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
"""

_JS = """
<script>
(function(){
  // exclusive coin accordions (fallback for browsers without <details name>)
  var coins=[].slice.call(document.querySelectorAll('details.coin'));
  coins.forEach(function(d){d.addEventListener('toggle',function(){
    if(d.open){coins.forEach(function(o){if(o!==d)o.open=false;});}
  });});

  // recent-log filters (coin / tf / result)
  var state={coin:'all',tf:'all',result:'all'};
  function applyFilters(){
    var rows=document.querySelectorAll('#log-table [data-row]');var shown=0;
    rows.forEach(function(r){
      var ok=(state.coin==='all'||r.getAttribute('data-coin')===state.coin)
        &&(state.tf==='all'||r.getAttribute('data-tf')===state.tf)
        &&(state.result==='all'||r.getAttribute('data-result')===state.result);
      r.style.display=ok?'':'none'; if(ok)shown++;
    });
    var e=document.querySelector('.filter-empty'); if(e)e.hidden=shown!==0;
  }
  document.querySelectorAll('.fbtn').forEach(function(b){
    b.addEventListener('click',function(){
      var f=b.getAttribute('data-f');state[f]=b.getAttribute('data-v');
      document.querySelectorAll('.fbtn[data-f="'+f+'"]').forEach(function(x){
        x.classList.toggle('active',x===b);});
      applyFilters();
    });
  });

  // confirmation modal for dangerous actions
  var modal=document.getElementById('modal');
  if(modal){
    var msg=document.getElementById('modal-msg');var pending=null;
    document.querySelectorAll('form[data-confirm]').forEach(function(f){
      f.addEventListener('submit',function(e){
        if(f.dataset.ok==='1')return;
        e.preventDefault();pending=f;msg.textContent=f.getAttribute('data-confirm');
        modal.hidden=false;
      });
    });
    document.getElementById('modal-ok').onclick=function(){
      if(pending){pending.dataset.ok='1';pending.submit();}};
    document.getElementById('modal-cancel').onclick=function(){modal.hidden=true;pending=null;};
    modal.addEventListener('click',function(e){if(e.target===modal){modal.hidden=true;pending=null;}});
  }
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

    tz = display_tz_label(offset_hours)
    generated = to_display_time(datetime.now(timezone.utc), offset_hours)
    refresh_meta = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds and refresh_seconds > 0
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Research dashboard for validating crypto next-candle prediction accuracy.">
{refresh_meta}
<title>TradeBotVol | Prediction Dashboard</title>
<style>{_CSS}</style></head>
<body>
<div class="shell">
  {_sidebar()}
  <main class="content">
    {_header(generated, refresh_seconds, tz)}
    {_research_notice()}
    {_verdict(resolved, report)}
    {_control_panel(controls_enabled, control_message)}
    {_settings_panel(controls_enabled, config_path)}
    {_how_to_read()}
    {_kpis(all_df, resolved, report)}
    {_validation_health(all_df, resolved, report)}
    {_rankings(report)}
    {_coin_accordion(coins, offset_hours)}
    {_recent_log(all_df, offset_hours)}
    <footer>Prediction-only research tool · times in {_esc(tz)} ·
    judge from large samples, not small ones.</footer>
  </main>
</div>
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
