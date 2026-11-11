#!/usr/bin/env bash
# Instala e sobe a API + frontend como serviço systemd na porta 80.
# Uso (dentro do EC2, Amazon Linux 2023 ou Ubuntu):
#   git clone https://github.com/Leofrancaa/ml_aws.git && cd ml_aws && sudo bash deploy/setup.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_USER="${SUDO_USER:-$(whoami)}"
PORT="${PORT:-80}"

# 1. uv (gerencia o Python 3.14 exigido pelo projeto, sem mexer no Python do sistema)
if ! sudo -u "$APP_USER" bash -lc 'command -v uv' >/dev/null 2>&1; then
  sudo -u "$APP_USER" bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
UV="$(sudo -u "$APP_USER" bash -lc 'command -v uv || echo $HOME/.local/bin/uv')"

# 2. Dependências exatamente como no uv.lock
sudo -u "$APP_USER" bash -lc "cd '$APP_DIR' && '$UV' sync --frozen --no-dev"

# 3. Serviço systemd (reinicia sozinho e sobe no boot)
cat > /etc/systemd/system/ml-api.service <<EOF
[Unit]
Description=API de Sentimento (FastAPI)
After=network-online.target
Wants=network-online.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/uvicorn app.api:app --host 0.0.0.0 --port $PORT --workers 2
AmbientCapabilities=CAP_NET_BIND_SERVICE
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now ml-api
# espera a API carregar o modelo (até ~60s em instância pequena)
for _ in $(seq 30); do curl -fsS "http://localhost:$PORT/health" && echo && break; sleep 2; done
echo "OK -> http://$(curl -fsS --max-time 2 http://checkip.amazonaws.com 2>/dev/null || echo '<IP-PUBLICO>'):$PORT"
