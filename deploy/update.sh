#!/usr/bin/env bash
#
# Push updated code to the already-running VM and restart the services.
# Run this in Cloud Shell from the freshly-unzipped project folder, AFTER
# uploading a new preduct-bot-deploy.zip:
#
#     bash deploy/update.sh
#
# It copies changed files, refreshes dependencies, (re)installs the limited
# sudo rule for the dashboard control buttons, and restarts both services.
set -euo pipefail

ZONE="${ZONE:-asia-northeast1-b}"
VM="${VM:-$(gcloud compute instances list --format='value(name)' | head -1)}"
REMOTE_DIR="~/preduct-bot"

echo "==> Updating ${VM} (${ZONE})..."
gcloud compute scp --recurse --zone "${ZONE}" \
  ./predictor ./deploy ./run_predictor.py ./train_model.py \
  ./backtest_predictions.py ./dashboard.py ./config.json ./requirements.txt \
  "${VM}:${REMOTE_DIR}/"

# Remote: refresh deps, (re)install the sudoers rule using the VM's own user,
# then restart the services. Single-quoted so $USER expands on the VM.
gcloud compute ssh "${VM}" --zone "${ZONE}" \
  --ssh-flag="-o StrictHostKeyChecking=no" --command '
set -e
cd ~/preduct-bot
~/preduct-bot/.venv/bin/pip install -q -r requirements.txt || true
printf "%s ALL=(root) NOPASSWD: /usr/bin/systemctl start preduct-predictor.service, /usr/bin/systemctl stop preduct-predictor.service, /usr/bin/systemctl restart preduct-predictor.service, /usr/sbin/shutdown\n" "$USER" \
  | sudo tee /etc/sudoers.d/preduct-control >/dev/null
sudo chmod 440 /etc/sudoers.d/preduct-control
sudo systemctl restart preduct-dashboard.service preduct-predictor.service
echo "---- service status ----"
systemctl is-active preduct-predictor.service preduct-dashboard.service
'
echo "==> Update complete."
