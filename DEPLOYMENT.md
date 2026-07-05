# Production deployment — https://tradebotvol.com

Serve the existing dashboard securely at **https://tradebotvol.com** (no
`:8080`) behind Nginx + Let's Encrypt TLS, with login protecting the dashboard
and its control actions.

Nothing about prediction, scoring, or data handling changes — this only puts a
reverse proxy, HTTPS, and a login gate in front of the app, which keeps running
internally on `127.0.0.1:8080`.

Current state: app already runs on the GCP VM `preduct-bot` (Tokyo) via the
`preduct-predictor` and `preduct-dashboard` systemd services, reachable at
`http://<VM-IP>:8080/`.

---

## 0. One-time: get the latest code onto the VM

If you changed code locally, redeploy first (see `deploy/redeploy.sh`). Then SSH in:

```bash
gcloud compute ssh preduct-bot --zone asia-northeast1-b
cd ~/preduct-bot
```

---

## 1. Reserve a static IP (so the IP never changes)

In Google Cloud Console: **VPC network → IP addresses → Reserve external static
address**, attach it to the `preduct-bot` VM. Or via CLI:

```bash
# promote the VM's current ephemeral IP to static
gcloud compute addresses create tradebotvol-ip \
  --addresses 35.221.114.67 --region asia-northeast1
```

Confirm the VM's external IP (use this in DNS):

```bash
gcloud compute instances describe preduct-bot --zone asia-northeast1-b \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

## 2. Point DNS at the VM

At your domain registrar / DNS provider for `tradebotvol.com`, add:

| Type | Name | Value            | TTL  |
|------|------|------------------|------|
| A    | `@`  | `35.221.114.67`  | 300  |
| A    | `www`| `35.221.114.67`  | 300  |

Wait for propagation, then verify from anywhere:

```bash
dig +short tradebotvol.com        # should print 35.221.114.67
dig +short www.tradebotvol.com    # should print 35.221.114.67
```

## 3. Google Cloud firewall — open 80 and 443

```bash
gcloud compute firewall-rules create allow-http-https \
  --allow tcp:80,tcp:443 --source-ranges 0.0.0.0/0 \
  --target-tags preduct-dashboard
```

The VM already has the `preduct-dashboard` network tag. After Nginx works you
can **close public 8080** (Nginx reaches it locally):

```bash
gcloud compute firewall-rules update allow-preduct-dashboard \
  --source-ranges 127.0.0.1/32     # effectively no public access to :8080
# (or delete it: gcloud compute firewall-rules delete allow-preduct-dashboard)
```

## 4. Turn on the dashboard login

Create `~/preduct-bot/.env` from the template and set a username + password hash:

```bash
cd ~/preduct-bot
cp -n .env.example .env
# generate a password hash (replace with a strong password):
.venv/bin/python dashboard.py --hash-password 'choose-a-strong-password'
# copy the printed hash, then edit .env:
nano .env
#   ADMIN_USERNAME=admin
#   ADMIN_PASSWORD_HASH=<the hash you just generated>
#   PORT=8080
sudo systemctl restart preduct-dashboard
```

Now the dashboard and every control route (pause / resume / restart / clear /
shutdown / settings) require this login. `GET /health` stays open for monitoring.
Leaving both fields blank keeps it open (local dev only).

## 5. Install Nginx and the site config

```bash
sudo apt update
sudo apt install nginx -y

sudo cp ~/preduct-bot/deploy/nginx/tradebotvol.com.conf \
        /etc/nginx/sites-available/tradebotvol.com
sudo ln -sf /etc/nginx/sites-available/tradebotvol.com \
            /etc/nginx/sites-enabled/tradebotvol.com
sudo rm -f /etc/nginx/sites-enabled/default   # remove the placeholder site

sudo nginx -t            # test config
sudo systemctl reload nginx
```

At this point `http://tradebotvol.com` should already load the dashboard
(through the proxy, still HTTP).

## 6. Enable HTTPS with Certbot

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d tradebotvol.com -d www.tradebotvol.com
```

Choose **redirect HTTP → HTTPS** when prompted. Certbot edits the Nginx config
to add the 443 server block and the redirect automatically.

Test auto-renewal (certs renew via a systemd timer):

```bash
sudo certbot renew --dry-run
```

## 7. Process manager (already systemd)

The app already runs under **systemd** (installed by `deploy/setup.sh`):

- `preduct-predictor.service` — the prediction loop
- `preduct-dashboard.service` — the web dashboard on `127.0.0.1:8080`

Both have `Restart=always` and are `enable`d, so they **start on reboot** and
**restart on crash**, and log to `~/preduct-bot/logs/`. Useful commands:

```bash
sudo systemctl status preduct-dashboard preduct-predictor
sudo systemctl restart preduct-dashboard        # after editing .env
journalctl -u preduct-dashboard -f              # follow logs
```

Nginx also runs under systemd and starts on boot.

---

## 8. Verify

```bash
curl -s https://tradebotvol.com/health           # {"status":"ok",...}
curl -sI http://tradebotvol.com                  # 301 -> https
```

In a browser:

1. Open **https://tradebotvol.com** → padlock shows a valid certificate.
2. A **login prompt** appears → enter ADMIN_USERNAME / password.
3. Dashboard loads, no `:8080` in the URL.
4. Live refresh still updates the numbers.
5. Prediction log, KPIs, coin accordions, rankings all load.
6. **Pause predictions** works after login.
7. **Clear log** shows the confirmation modal, then works.
8. **Shut down** shows the confirmation modal, then works.
9. `https://www.tradebotvol.com` works too.

## Rollback / notes

- To disable login later: blank `ADMIN_USERNAME`/`ADMIN_PASSWORD_HASH` in `.env`
  and `sudo systemctl restart preduct-dashboard`.
- To bypass Nginx temporarily, re-open firewall `tcp:8080` and hit the IP.
- HTTPS certs auto-renew; `certbot renew --dry-run` verifies the pipeline.
- The app is Python (systemd), not Node — `pm2`/`NODE_ENV` are not used.
