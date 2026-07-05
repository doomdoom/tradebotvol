# Deploying the prediction bot to Google Cloud (24/7 + remote dashboard)

This runs the predictor **and** the live dashboard on an always-on Google
Cloud VM, so the dashboard auto-updates and is reachable from anywhere
(phone, laptop) at `http://<VM_IP>:8080/`.

---

## ⚠️ Read this first — Binance blocks US IP addresses

Binance's market-data API returns **HTTP 451 ("restricted location")** from US
IPs. Google Cloud's free `e2-micro` tier is **US-only**, so it will **not**
work. You must deploy in a **non-US region**. The scripts default to **Tokyo
(`asia-northeast1`)**. Other known-good regions: Singapore
(`asia-southeast1`), Frankfurt (`europe-west3`), Sydney (`australia-southeast1`).

Rough cost: an `e2-small` runs about **$13/month** if left on 24/7 (an
`e2-micro` in a non-US region is ~$6–8/mo but tighter on RAM). **Stop or delete
the VM when you're done** to stop billing (commands at the bottom).

---

## Why a VM (Compute Engine) and not Cloud Run / App Engine?

The predictor is a **continuous loop** that must run all the time and shares a
SQLite file with the dashboard. Cloud Run and App Engine are request-driven and
scale to zero — they'd kill the loop. A small Compute Engine VM is the right
fit and the simplest to reason about.

---

## Option A — automated (one script)

From **this project folder on your machine**, with the
[gcloud CLI](https://cloud.google.com/sdk/docs/install) installed:

```bash
# 1. Authenticate and pick your billing-enabled project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Provision the VM, firewall, copy code, and start the services.
#    MY_IP locks the dashboard to your current public IP (recommended).
MY_IP=$(curl -s ifconfig.me) bash deploy/provision_gcp.sh
```

When it finishes it prints the dashboard URL, e.g.
`http://34.85.x.x:8080/`. Open it — it auto-refreshes every 3 seconds and the
numbers climb as candles close.

Tweak defaults with env vars, e.g. a different region/port:

```bash
ZONE=asia-southeast1-b MACHINE_TYPE=e2-small DASHBOARD_PORT=8080 \
  MY_IP=$(curl -s ifconfig.me) bash deploy/provision_gcp.sh
```

---

## Option B — manual (understand each step)

```bash
# 1. Create the VM in a NON-US zone, tagged for the firewall rule
gcloud compute instances create preduct-bot \
  --zone asia-northeast1-b \
  --machine-type e2-small \
  --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 20GB \
  --tags preduct-dashboard

# 2. Allow the dashboard port ONLY from your IP (safer than the whole internet)
gcloud compute firewall-rules create allow-preduct-dashboard \
  --allow tcp:8080 \
  --source-ranges "$(curl -s ifconfig.me)/32" \
  --target-tags preduct-dashboard

# 3. Copy the code up
gcloud compute ssh preduct-bot --zone asia-northeast1-b --command "mkdir -p ~/preduct-bot"
gcloud compute scp --recurse --zone asia-northeast1-b \
  ./predictor ./deploy ./run_predictor.py ./train_model.py \
  ./backtest_predictions.py ./dashboard.py ./config.json ./requirements.txt \
  preduct-bot:~/preduct-bot/

# 4. Install deps + start the 24/7 services
gcloud compute ssh preduct-bot --zone asia-northeast1-b \
  --command "cd ~/preduct-bot && bash deploy/setup.sh"

# 5. Get the public IP and open http://<IP>:8080/
gcloud compute instances describe preduct-bot --zone asia-northeast1-b \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

---

## Managing the running bot

```bash
gcloud compute ssh preduct-bot --zone asia-northeast1-b   # log in

# follow live output
journalctl -u preduct-predictor -f
journalctl -u preduct-dashboard -f

# restart after editing config.json (re-copy it first with scp)
sudo systemctl restart preduct-predictor preduct-dashboard

# check status
systemctl status preduct-predictor preduct-dashboard
```

Both services have `Restart=always` and are enabled at boot, so they survive
crashes and VM reboots.

---

## Security notes

* The dashboard has **no login**. Keep the firewall `--source-ranges` limited
  to your own IP. Only widen to `0.0.0.0/0` if you accept that anyone with the
  URL can view your predictions (it's read-only market data — no keys, no
  trading — but still).
* The bot needs **no Binance API key** (public data only), so there are no
  secrets to leak on the VM.
* If your home IP changes, update the rule:
  ```bash
  gcloud compute firewall-rules update allow-preduct-dashboard \
    --source-ranges "$(curl -s ifconfig.me)/32"
  ```

---

## Stop paying when finished

```bash
# pause (keeps the disk, stops compute billing)
gcloud compute instances stop preduct-bot --zone asia-northeast1-b

# or remove everything
gcloud compute instances delete preduct-bot --zone asia-northeast1-b
gcloud compute firewall-rules delete allow-preduct-dashboard
```
