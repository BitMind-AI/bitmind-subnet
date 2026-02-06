#!/usr/bin/env bash
# Entrypoint for the bitmind-subnet validator Docker container.
# Translates environment variables into CLI arguments and generates the
# supervisord configuration for the 3 validator processes.

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
: "${WALLET_NAME:=default}"
: "${WALLET_HOTKEY:=default}"
: "${CHAIN_ENDPOINT:=wss://entrypoint-finney.opentensor.ai:443}"
: "${CALLBACK_PORT:=10525}"
: "${DEVICE:=cuda}"
: "${LOGLEVEL:=info}"
: "${BENCHMARK_API_URL:=https://gas.bitmind.ai}"
: "${SCRAPER_INTERVAL:=300}"
: "${DATASET_INTERVAL:=1800}"
: "${HEARTBEAT:=false}"
: "${START_VALIDATOR:=true}"
: "${START_GENERATOR:=true}"
: "${START_DATA:=true}"
: "${WANDB_API_KEY:=}"

# ── Derive netuid from chain endpoint ───────────────────────────────────────
# Mirrors validator.config.js getNetworkSettings()
if [[ -z "${NETUID:-}" ]]; then
    if [[ "$CHAIN_ENDPOINT" == *"test"* ]]; then
        NETUID=379
    elif [[ "$CHAIN_ENDPOINT" == *"finney"* ]]; then
        NETUID=34
    else
        echo "ERROR: Cannot derive NETUID from CHAIN_ENDPOINT=$CHAIN_ENDPOINT"
        echo "       Set NETUID explicitly in your .env.validator file."
        exit 1
    fi
fi

# ── Map log level to CLI param ──────────────────────────────────────────────
# Mirrors validator.config.js getLogParam()
case "${LOGLEVEL,,}" in
    trace) LOG_PARAM="--logging.trace" ;;
    debug) LOG_PARAM="--logging.debug" ;;
    *)     LOG_PARAM="--logging.info"  ;;
esac

# ── Build validator command ─────────────────────────────────────────────────
VALIDATOR_CMD="/app/.venv/bin/python neurons/validator/validator.py"
VALIDATOR_CMD+=" --wallet.name ${WALLET_NAME}"
VALIDATOR_CMD+=" --wallet.hotkey ${WALLET_HOTKEY}"
VALIDATOR_CMD+=" --netuid ${NETUID}"
VALIDATOR_CMD+=" --subtensor.chain_endpoint ${CHAIN_ENDPOINT}"
VALIDATOR_CMD+=" --neuron.callback_port ${CALLBACK_PORT}"
VALIDATOR_CMD+=" --cache.base-dir /root/.cache/sn34"
VALIDATOR_CMD+=" --benchmark.api-url ${BENCHMARK_API_URL}"
VALIDATOR_CMD+=" ${LOG_PARAM}"
VALIDATOR_CMD+=" --autoupdate-off"

if [[ -n "${EXTERNAL_CALLBACK_PORT:-}" ]]; then
    VALIDATOR_CMD+=" --neuron.external-callback-port ${EXTERNAL_CALLBACK_PORT}"
fi

if [[ "${HEARTBEAT,,}" == "true" ]]; then
    VALIDATOR_CMD+=" --heartbeat"
fi

# ── Build generator command ─────────────────────────────────────────────────
GENERATOR_CMD="/app/.venv/bin/python neurons/validator/services/generator_service.py"
GENERATOR_CMD+=" --wallet.name ${WALLET_NAME}"
GENERATOR_CMD+=" --wallet.hotkey ${WALLET_HOTKEY}"
GENERATOR_CMD+=" --cache.base-dir /root/.cache/sn34"
GENERATOR_CMD+=" --device ${DEVICE}"
GENERATOR_CMD+=" --log-level ${LOGLEVEL}"

# ── Build data service command ──────────────────────────────────────────────
DATA_CMD="/app/.venv/bin/python neurons/validator/services/data_service.py"
DATA_CMD+=" --wallet.name ${WALLET_NAME}"
DATA_CMD+=" --wallet.hotkey ${WALLET_HOTKEY}"
DATA_CMD+=" --netuid ${NETUID}"
DATA_CMD+=" --subtensor.chain_endpoint ${CHAIN_ENDPOINT}"
DATA_CMD+=" --cache.base-dir /root/.cache/sn34"
DATA_CMD+=" --scraper-interval ${SCRAPER_INTERVAL}"
DATA_CMD+=" --dataset-interval ${DATASET_INTERVAL}"
DATA_CMD+=" ${LOG_PARAM}"

# ── HuggingFace env ─────────────────────────────────────────────────────────
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

# ── Generate supervisord.conf ──────────────────────────────────────────────
CONF=/etc/supervisor/conf.d/supervisord.conf

cat > "$CONF" <<SUPERVISORD_EOF
[supervisord]
nodaemon=true
user=root
logfile=/dev/null
logfile_maxbytes=0
pidfile=/var/run/supervisord.pid

[program:sn34-validator]
command=${VALIDATOR_CMD}
directory=/app
autostart=${START_VALIDATOR}
autorestart=true
startretries=10
startsecs=10
stopwaitsecs=30
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0
environment=WANDB_API_KEY="${WANDB_API_KEY}"

[program:sn34-generator]
command=${GENERATOR_CMD}
directory=/app
autostart=${START_GENERATOR}
autorestart=true
startretries=10
startsecs=10
stopwaitsecs=30
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0

[program:sn34-data]
command=${DATA_CMD}
directory=/app
autostart=${START_DATA}
autorestart=true
startretries=10
startsecs=10
stopwaitsecs=30
stdout_logfile=/dev/fd/1
stdout_logfile_maxbytes=0
stderr_logfile=/dev/fd/2
stderr_logfile_maxbytes=0
environment=TMPDIR="/root/.cache/sn34/tmp",TEMP="/root/.cache/sn34/tmp",TMP="/root/.cache/sn34/tmp"
SUPERVISORD_EOF

# ── Ensure cache directories exist ─────────────────────────────────────────
mkdir -p /root/.cache/sn34/tmp /root/.cache/huggingface

# ── Start Xvfb for headless Chrome (used by data service scraper) ──────────
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp &
export DISPLAY=:99

echo "========================================"
echo "  bitmind-subnet Validator (Docker)"
echo "========================================"
echo "  WALLET:     ${WALLET_NAME}/${WALLET_HOTKEY}"
echo "  NETUID:     ${NETUID}"
echo "  CHAIN:      ${CHAIN_ENDPOINT}"
echo "  DEVICE:     ${DEVICE}"
echo "  CALLBACK:   ${CALLBACK_PORT}"
echo "  LOGLEVEL:   ${LOGLEVEL}"
echo "  VALIDATOR:  ${START_VALIDATOR}"
echo "  GENERATOR:  ${START_GENERATOR}"
echo "  DATA:       ${START_DATA}"
echo "========================================"

# ── Launch supervisord (foreground) ─────────────────────────────────────────
exec /usr/bin/supervisord -c "$CONF"
