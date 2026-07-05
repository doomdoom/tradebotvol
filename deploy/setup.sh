#!/usr/bin/env bash
#
# One-shot provisioner for the prediction bot on a Linux VM (Debian/Ubuntu).
#
# Run it from the project root on the VM, AFTER copying the code across:
#     cd ~/preduct-bot && bash deploy/setup.sh
#
# It creates a virtualenv, installs dependencies, and installs + starts two
# systemd services that run 24/7 and restart on crash or reboot:
#     preduct-predictor   -> run_predictor.py     (generates predictions)
#     preduct-dashboard   -> dashboard.py --serve (live web dashboard)
#
# Override the dashboard port / refresh via environment variables:
#     DASHBOARD_PORT=8080 DASHBOARD_REFRESH=3 bash deploy/setup.sh
set -euo pipefail

# --- resolve paths and identity -------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_USER="$(whoami)"
PY_BIN="${APP_DIR}/.venv/bin/python"
PIP_BIN="${APP_DIR}/.venv/bin/pip"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
DASHBOARD_REFRESH="${DASHBOARD_REFRESH:-3}"

echo "==> App dir : ${APP_DIR}"
echo "==> User    : ${RUN_USER}"
echo "==> Port    : ${DASHBOARD_PORT} (refresh ${DASHBOARD_REFRESH}s)"

# --- system packages -------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing OS packages (python3, venv, pip)..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3 python3-venv python3-pip
fi

# --- python environment ----------------------------------------------------
echo "==> Creating virtualenv and installing requirements..."
python3 -m venv "${APP_DIR}/.venv"
"${PIP_BIN}" install --upgrade pip -q
"${PIP_BIN}" install -r "${APP_DIR}/requirements.txt" -q

mkdir -p "${APP_DIR}/data" "${APP_DIR}/models" "${APP_DIR}/logs"

# --- systemd unit: predictor ----------------------------------------------
echo "==> Installing systemd services..."
sudo tee /etc/systemd/system/preduct-predictor.service >/dev/null <<UNIT
[Unit]
Description=Preduct bot - next-candle predictor loop
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${PY_BIN} ${APP_DIR}/run_predictor.py --config ${APP_DIR}/config.json
Restart=always
RestartSec=10
StandardOutput=append:${APP_DIR}/logs/predictor.stdout.log
StandardError=append:${APP_DIR}/logs/predictor.stderr.log

[Install]
WantedBy=multi-user.target
UNIT

# --- systemd unit: dashboard ----------------------------------------------
sudo tee /etc/systemd/system/preduct-dashboard.service >/dev/null <<UNIT
[Unit]
Description=Preduct bot - live web dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${APP_DIR}
# Optional .env supplies ADMIN_USERNAME / ADMIN_PASSWORD_HASH / PORT (the leading
# '-' makes it optional so the service still starts without a .env file).
EnvironmentFile=-${APP_DIR}/.env
ExecStart=${PY_BIN} ${APP_DIR}/dashboard.py --serve --no-open \\
  --host 0.0.0.0 --port ${DASHBOARD_PORT} --refresh ${DASHBOARD_REFRESH}
Restart=always
RestartSec=10
StandardOutput=append:${APP_DIR}/logs/dashboard.stdout.log
StandardError=append:${APP_DIR}/logs/dashboard.stderr.log

[Install]
WantedBy=multi-user.target
UNIT

# --- allow the dashboard to control the predictor + power off --------------
# The dashboard's pause/resume/shutdown buttons need to run these specific
# privileged commands without a password. Scope is limited to exactly these.
echo "==> Granting the dashboard limited sudo for control buttons..."
sudo tee /etc/sudoers.d/preduct-control >/dev/null <<SUDO
${RUN_USER} ALL=(root) NOPASSWD: /usr/bin/systemctl start preduct-predictor.service, /usr/bin/systemctl stop preduct-predictor.service, /usr/bin/systemctl restart preduct-predictor.service, /usr/sbin/shutdown
SUDO
sudo chmod 440 /etc/sudoers.d/preduct-control

# --- enable + start --------------------------------------------------------
sudo systemctl daemon-reload
sudo systemctl enable --now preduct-predictor.service preduct-dashboard.service

echo ""
echo "==> Done. Service status:"
sudo systemctl --no-pager --lines=0 status preduct-predictor preduct-dashboard || true
echo ""
echo "Dashboard will be reachable on port ${DASHBOARD_PORT} once the GCP"
echo "firewall rule is in place. Follow logs with:"
echo "    journalctl -u preduct-predictor -f"
echo "    journalctl -u preduct-dashboard -f"
