"""Generate a static HTML dashboard from the prediction log.

Reads data/predictions.db (or the paths in config.json) and writes a single
self-contained dashboard.html: summary cards, per-pair accuracy, confusion
matrices, confidence-vs-accuracy, rankings and the latest predictions. No
server, no JavaScript, no external assets — just open the file in a browser.

Usage:
    py dashboard.py                       # writes dashboard.html and opens it
    py dashboard.py --output report.html  # custom output path
    py dashboard.py --no-open             # just write the file
    py dashboard.py --serve               # LIVE: auto-refreshing web server
    py dashboard.py --serve --refresh 2   # refresh every 2 seconds
    py dashboard.py --hash-password PW    # print a password hash for .env

Environment variables (loaded from .env via config):
    PORT                 default server port for --serve (default 8787)
    ADMIN_USERNAME       enable login: the dashboard username
    ADMIN_PASSWORD_HASH  sha256 hex of the password (see --hash-password)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import webbrowser
from pathlib import Path

from predictor.config import load_config
from predictor.dashboard import serve_dashboard, write_dashboard
from predictor.logger import setup_logging
from predictor.storage import PredictionStorage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="path to config.json")
    parser.add_argument("--output", default="dashboard.html", help="output HTML path")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="run a live auto-refreshing web server instead of writing a file",
    )
    parser.add_argument("--host", default="0.0.0.0", help="server host (--serve)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8787")),
        help="server port (--serve; falls back to $PORT then 8787)",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="auto-refresh interval in seconds (--serve; min 1)",
    )
    parser.add_argument(
        "--hash-password",
        metavar="PASSWORD",
        help="print the sha256 hash of PASSWORD (for ADMIN_PASSWORD_HASH) and exit",
    )
    args = parser.parse_args()

    if args.hash_password:
        print(hashlib.sha256(args.hash_password.encode("utf-8")).hexdigest())
        return 0

    config = load_config(args.config)  # also loads .env into os.environ
    setup_logging(config.log_level)

    # In file mode a missing DB is fatal; in serve mode we start anyway and
    # show an empty dashboard that fills in as the predictor writes rows.
    if not Path(config.db_path).exists() and not args.serve:
        print(
            f"No prediction database found at {config.db_path}.\n"
            "Run 'py run_predictor.py' or 'py backtest_predictions.py --save' first "
            "to generate predictions."
        )
        return 1

    if args.serve:
        if not Path(config.db_path).exists():
            print(
                f"Note: {config.db_path} does not exist yet — serving an empty "
                "dashboard until run_predictor.py starts writing predictions."
            )
        url = f"http://{args.host}:{args.port}/"
        if not args.no_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        auth_user = os.getenv("ADMIN_USERNAME") or None
        auth_hash = os.getenv("ADMIN_PASSWORD_HASH") or None
        # PUBLIC_VIEW=true -> anyone can view; control buttons still need login.
        public_view = os.getenv("PUBLIC_VIEW", "").strip().lower() in ("1", "true", "yes")
        serve_dashboard(
            config.db_path,
            config.csv_path,
            host=args.host,
            port=args.port,
            refresh_seconds=max(1, args.refresh),
            offset_hours=config.display_utc_offset_hours,
            controls_enabled=config.dashboard_controls_enabled,
            config_path=args.config,
            auth_user=auth_user,
            auth_password_hash=auth_hash,
            protect_view=not public_view,
            market_type=config.market_type,
            allowed_symbols=config.symbols,
        )
        return 0

    storage = PredictionStorage(
        config.db_path, config.csv_path, sqlite_enabled=True, csv_enabled=False
    )
    try:
        out = write_dashboard(
            storage, args.output, offset_hours=config.display_utc_offset_hours
        )
    finally:
        storage.close()

    print(f"Dashboard written to: {out}")
    if not args.no_open:
        webbrowser.open(Path(out).as_uri())
        print("Opened in your default browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
