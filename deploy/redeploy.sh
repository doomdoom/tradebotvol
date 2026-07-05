#!/usr/bin/env bash
#
# Robust (re)deploy to an existing VM. Run in Cloud Shell from the unzipped
# project folder:
#
#     SOURCE_RANGE=1.2.3.4/32 bash deploy/redeploy.sh
#
# It auto-discovers the VM and dashboard firewall rule (so it does not matter
# whether earlier resources were named product-* or preduct-*), copies the
# latest code, stops any previously-installed services under either name, and
# reinstalls a single clean, consistently-named set via setup.sh.
set -euo pipefail

ZONE="${ZONE:-asia-northeast1-b}"
VM="${VM:-$(gcloud compute instances list --format='value(name)' --filter='name~bot' | head -1)}"
if [ -z "${VM}" ]; then
  echo "ERROR: no VM found whose name contains 'bot'. Set VM=<name> and retry." >&2
  exit 1
fi
echo "==> Target VM: ${VM} (${ZONE})"

# Optionally lock the dashboard firewall to a source range (recommended).
if [ -n "${SOURCE_RANGE:-}" ]; then
  RULE="$(gcloud compute firewall-rules list --format='value(name)' \
          --filter='name~dashboard' | head -1)"
  if [ -n "${RULE}" ]; then
    echo "==> Locking firewall '${RULE}' to ${SOURCE_RANGE}"
    gcloud compute firewall-rules update "${RULE}" --source-ranges "${SOURCE_RANGE}" || true
  fi
fi

echo "==> Copying latest code to the VM..."
gcloud compute ssh "${VM}" --zone "${ZONE}" \
  --ssh-flag="-o StrictHostKeyChecking=no" --command "mkdir -p ~/preduct-bot"
gcloud compute scp --recurse --zone "${ZONE}" \
  ./predictor ./deploy ./run_predictor.py ./train_model.py \
  ./backtest_predictions.py ./dashboard.py ./config.json ./requirements.txt \
  "${VM}:~/preduct-bot/"

echo "==> Stopping any old services and installing the clean set..."
gcloud compute ssh "${VM}" --zone "${ZONE}" \
  --ssh-flag="-o StrictHostKeyChecking=no" --command '
set -e
for svc in product-predictor product-dashboard preduct-predictor preduct-dashboard; do
  sudo systemctl stop "$svc.service" 2>/dev/null || true
  sudo systemctl disable "$svc.service" 2>/dev/null || true
done
cd ~/preduct-bot
bash deploy/setup.sh
echo "---- active services ----"
systemctl is-active preduct-predictor.service preduct-dashboard.service
'
echo "==> Redeploy complete. Dashboard: http://<VM-external-IP>:8080/"
