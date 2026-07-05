#!/usr/bin/env bash
#
# Provision a Google Cloud VM for the prediction bot and deploy the code.
# Run this on YOUR machine (where gcloud is installed and authenticated).
#
# Prerequisites:
#   * gcloud CLI installed and logged in:   gcloud auth login
#   * a billing-enabled project:            gcloud config set project YOUR_PROJECT
#
# IMPORTANT — Binance geo-blocks US IPs (HTTP 451). Keep ZONE in a non-US
# region (Tokyo/Singapore/Frankfurt). The US-only free tier will NOT work.
#
# Usage:
#   MY_IP=$(curl -s ifconfig.me) ./deploy/provision_gcp.sh
set -euo pipefail

# --- configurable knobs (override via environment) ------------------------
VM_NAME="${VM_NAME:-preduct-bot}"
ZONE="${ZONE:-asia-northeast1-b}"          # Tokyo. Non-US on purpose.
MACHINE_TYPE="${MACHINE_TYPE:-e2-small}"    # 2 vCPU burst, 2 GB RAM (~$13/mo)
IMAGE_FAMILY="${IMAGE_FAMILY:-debian-12}"
IMAGE_PROJECT="${IMAGE_PROJECT:-debian-cloud}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"
NET_TAG="preduct-dashboard"
# Restrict who can reach the dashboard. Default: only your current public IP.
# Set SOURCE_RANGE=0.0.0.0/0 to allow the whole internet (NOT recommended;
# the dashboard has no login).
SOURCE_RANGE="${SOURCE_RANGE:-${MY_IP:-}/32}"

if [[ "${SOURCE_RANGE}" == "/32" ]]; then
  echo "ERROR: set MY_IP (e.g. MY_IP=\$(curl -s ifconfig.me)) or SOURCE_RANGE." >&2
  exit 1
fi

echo "==> Creating VM ${VM_NAME} in ${ZONE} (${MACHINE_TYPE})..."
gcloud compute instances create "${VM_NAME}" \
  --zone "${ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --image-family "${IMAGE_FAMILY}" \
  --image-project "${IMAGE_PROJECT}" \
  --boot-disk-size 20GB \
  --tags "${NET_TAG}"

echo "==> Creating firewall rule (tcp:${DASHBOARD_PORT} from ${SOURCE_RANGE})..."
gcloud compute firewall-rules create "allow-${NET_TAG}" \
  --allow "tcp:${DASHBOARD_PORT}" \
  --source-ranges "${SOURCE_RANGE}" \
  --target-tags "${NET_TAG}" \
  || echo "   (firewall rule already exists — skipping)"

echo "==> Waiting for SSH to come up..."
until gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "true" 2>/dev/null; do
  sleep 5
done

echo "==> Copying project to the VM..."
# Copy everything except local virtualenvs, caches and the local prediction DB.
TMP_LIST="$(mktemp)"
trap 'rm -f "${TMP_LIST}"' EXIT
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command "mkdir -p ~/preduct-bot"
gcloud compute scp --recurse --zone "${ZONE}" \
  ./predictor ./deploy \
  ./run_predictor.py ./train_model.py ./backtest_predictions.py ./dashboard.py \
  ./config.json ./requirements.txt ./README.md \
  "${VM_NAME}:~/preduct-bot/"

echo "==> Running remote setup (installs deps + starts 24/7 services)..."
gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --command \
  "cd ~/preduct-bot && DASHBOARD_PORT=${DASHBOARD_PORT} bash deploy/setup.sh"

EXTERNAL_IP="$(gcloud compute instances describe "${VM_NAME}" --zone "${ZONE}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

echo ""
echo "==================================================================="
echo " Deployed. Live dashboard:  http://${EXTERNAL_IP}:${DASHBOARD_PORT}/"
echo "==================================================================="
echo " Manage the VM:"
echo "   gcloud compute ssh ${VM_NAME} --zone ${ZONE}"
echo "   journalctl -u preduct-predictor -f      # follow predictions"
echo "   sudo systemctl restart preduct-predictor preduct-dashboard"
echo " Stop billing when done:"
echo "   gcloud compute instances stop ${VM_NAME} --zone ${ZONE}"
echo "   gcloud compute instances delete ${VM_NAME} --zone ${ZONE}"
