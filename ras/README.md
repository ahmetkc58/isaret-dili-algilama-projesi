# Raspberry Pi Deployment (ras)

This folder is a self-contained Raspberry Pi package for running `video_dashboard.py`.

## Files
- `setup_pi.sh`: Installs system packages and Python dependencies.
- `requirements-pi.txt`: Raspberry Pi oriented Python dependency list.
- `.env.pi.example`: Example environment configuration.
- `env.pi.example`: Same config template without hidden-dot filename.
- `run_dashboard_pi.sh`: Starts dashboard with Pi-friendly defaults.
- `sessizkelimeler-dashboard.service`: Optional systemd service.

## 1) Install Dependencies
```bash
cd /home/pi/sessizkelimeler/ras
chmod +x setup_pi.sh
./setup_pi.sh
```

## 2) Configure Paths And Tunings
```bash
cd /home/pi/sessizkelimeler/ras
cp env.pi.example env.pi
nano env.pi
```

Set at least these values:
- `MODEL_PATH`
- `WORDLIST_PATH`
- `VIDEO_PATH`

`run_dashboard_pi.sh` reads config from `.env.pi` first, then `env.pi`.

## 3) Run Manually
```bash
cd /home/pi/sessizkelimeler/ras
chmod +x run_dashboard_pi.sh
./run_dashboard_pi.sh
```

Dashboard URL:
- `http://127.0.0.1:8010`
- `http://<PI_IP>:8010`

## 4) Run As A Service (Optional)
```bash
sudo cp /home/pi/sessizkelimeler/ras/sessizkelimeler-dashboard.service /etc/systemd/system/
sudo nano /etc/systemd/system/sessizkelimeler-dashboard.service
sudo systemctl daemon-reload
sudo systemctl enable sessizkelimeler-dashboard
sudo systemctl start sessizkelimeler-dashboard
```

If your Raspberry Pi user is not `pi`, update `User=` in the service file.

Check status/logs:
```bash
sudo systemctl status sessizkelimeler-dashboard
journalctl -u sessizkelimeler-dashboard -f
```
