# Validator Guide

Run the SN34 validator (validator + generator + data services). Two ways: **PM2** (native) or **Docker**.

Prerequisites: [Installation Guide](Installation.md), Bittensor wallet, GPU for generation.

---

## Quick start: PM2

```bash
cp .env.validator.template .env.validator
# Edit .env.validator: WALLET_NAME, WALLET_HOTKEY, API keys, CHAIN_ENDPOINT
source .venv/bin/activate
gascli validator start
```

That starts three PM2 processes: `sn34-validator`, `sn34-generator`, `sn34-data`. Use `gascli v stop|status|logs|--help` as needed.

---

## Quick start: Docker

Three containers (validator, generator, data) share one image and volumes; one process per container.

```bash
cp .env.validator.template .env.validator
# Edit .env.validator (same as above; WALLET_PATH is for host wallet bind-mount)
docker compose --env-file .env.validator up -d --build
docker compose logs -f validator   # or generator, data
```

Always use `--env-file .env.validator`. First run downloads 100+ GB of models into the `hf-cache` volume.

**Update:** Manual: `git pull && docker compose --env-file .env.validator build && docker compose --env-file .env.validator up -d`. Automatic: cron with `docker/autoupdate.sh` (see [.env.validator.template](../.env.validator.template) comments for crontab example).

---

## Config

One file for both PM2 and Docker: `.env.validator`. Copy from [.env.validator.template](../.env.validator.template) and fill in wallet, API keys, `CHAIN_ENDPOINT`. The template lists every option and Docker-specific notes (e.g. `WALLET_PATH`, cache paths, autoupdate cron).

---

## Reference

| Path   | Start | Stop / logs |
|--------|-------|-------------|
| PM2    | `gascli validator start` | `gascli v stop` / `gascli v logs` |
| Docker | `docker compose --env-file .env.validator up -d` | `docker compose down` / `docker compose logs -f validator` (or `generator`, `data`) |
