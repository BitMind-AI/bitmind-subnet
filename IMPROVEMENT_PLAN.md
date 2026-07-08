# bitmind-subnet Improvement Plan

*Generated from a full-codebase audit, 2026-07-08. ~28k lines of Python audited across `gas/`, `neurons/`, tests, packaging, and ops tooling.*

The short version: the code works, but it works the way a Jenga tower works. This plan is ordered so that each phase pays for the next — emergencies first, then cheap correctness wins, then the structural surgery. Phases 0–2 are days of work. Phases 3–5 are the real investment.

---

## Phase 0 — Emergencies (do today)

### 0.1 Close the API-key exposure window — DONE
- `openrouter.env` contained a live OpenRouter key and was not gitignored; `.env.gen_miner` holds the same key plus a Runway key. Git history verified clean (pickaxe search found nothing) — keys were never committed, so no rotation needed and the files stay.
- ✅ Fixed 2026-07-08: `.gitignore` now has `*.env` / `.env.*` catch-alls with `!*.env.template` negations; `openrouter.env` verified ignored, templates verified still tracked.

### 0.2 Remaining .gitignore cleanup
- Add editor litter patterns: `#*#`, `.#*`, `*~`.
- Dedupe the duplicated blocks (`*.onnx`, `downloaded_models/*` each appear twice) and the now-redundant per-name `.env.*` entries.

### 0.3 Take out the trash (all untracked working-tree litter)
Delete: root `cli.py` (byte-identical copy of `gas/cli.py`, 1,129 lines), `#VEr#` (empty emacs artifact), `neurons/validator/#validator.py#` (18KB stale emacs autosave of the validator), `verification_ideass.txt` (0 bytes), `pr_description.md`. Triage: `verification_ideass.md`, `docs/c2pa_model_status.md`, `docs/verification_redesign_tifa.md` (commit or archive), `check_*_c2pa.py` + `dump_c2pa.py` (move to `scripts/` or delete), `install-macOS.sh` (commit it or kill it).

---

## Phase 1 — Correctness bugs (quick, high-value fixes)

These are live bugs found by inspection. Each is a small, independently shippable fix. Suggested order: security-adjacent first.

| # | Bug | Location |
|---|-----|----------|
| 1 | Non-deterministic on-chain model hash: `str(hash(model_hash))` uses Python's process-salted `hash()` — differs across runs/hosts | `neurons/discriminator/push_model.py:239` |
| 2 | SQL string interpolation of unvalidated `modality` and `LIMIT` (sibling method parameterizes correctly — one path is unsafe) | `gas/cache/db/media_store.py:374-377` |
| 3 | Wrong-table DELETE: cleanup deletes from `prompts` using **media** UUIDs | `gas/cache/db/media_store.py:720` |
| 4 | Mask retrieval always silently fails: `with_suffix("_mask.npy")` raises `ValueError` (needs leading dot), swallowed by outer except | `gas/cache/media_storage.py:338` |
| 5 | ~~Google scraper limit bug~~ / ~~`download_images(urls=...)` NameError~~ — moot: `gas/scraping/` had zero importers and was **deleted** (with its selenium/stamina deps and Chrome install blocks) | done 2026-07-08 |
| 7 | `on_block_interval` logs error on `None` interval then proceeds to `block % None` → `TypeError` (missing `return`) | `gas/utils/utils.py:52-56` |
| 8 | C2PA trust via bidirectional substring matching — a cert subject containing "Google"/"Adobe" passes a security boundary | `gas/verification/c2pa_verification.py:761-768, 800-821` |
| 9 | DB connection not closed on exception (`close()` not in `finally`) | `gas/cache/db/connection.py:19-36` |
| 10 | Webhook retry backoff exponentiates the delay itself: `retry_delay ** attempt` instead of `retry_delay * 2**attempt` | `gas/protocol/webhooks.py:447` |
| 11 | `WebhookStatsTracker` mutates shared dicts from per-request threads with no lock (lock only guards file writes) | `gas/protocol/webhooks.py:111, 402, 420` |
| 12 | Broken logging: printf-style `%s` args passed to loguru-style `bt.logging` — values never interpolated | `gas/protocol/gas_api_validator.py:35, 85, 89, 104` |
| 13 | Non-f-string error message emits literal `{model_name}`; unreachable `return {}` after `raise` two lines later | `gas/generation/media/generation_pipeline.py:242-248` |
| 14 | `clip_batch_size` parameter plumbed through and silently ignored (hardcoded 128/32 inside) | `gas/verification/verification_pipeline.py:95, 173, 184` |
| 15 | Autoupdater infinite loop if repo dir isn't named exactly `bitmind-subnet` (walks parents to `/` forever); unguarded `os.system("git pull")` with no return-code check | `gas/utils/autoupdater.py:204-207` |
| 16 | `validator.py` class-level mutable state: `block_callbacks`, `exit_context`, substrate handles are **class attributes** — callbacks accumulate across instantiations | `neurons/base.py:35-39` |
| 17 | `update_scores` mutates `self.scores` outside `_state_lock` while `save_state` takes it — racy | `neurons/validator/validator.py:329-338` |
| 18 | Epistula replay window one-sided: stale timestamps rejected, future-dated pass | `gas/protocol/epistula.py:82` |
| 19 | Dockerfile installs unpinned `diffusers` git master over the `0.36.0` pin in pyproject — Docker validators run different code than bare-metal | `Dockerfile` build stage vs `pyproject.toml:31` |
| 20 | `encoding.py` docstring claims PNG/AVI mime types; code returns JPEG/MP4. `assert` used for input validation (stripped under `-O`) | `gas/protocol/encoding.py:50, 66, 138-153` |

---

## Phase 2 — Hygiene & tooling (make the floor stop being lava)

### 2.1 CI (there is none)
- Add `.github/workflows/ci.yml`: ruff (lint + format check), pytest, and an import-smoke test (`python -c "import gas, neurons.validator.validator"` — this alone would catch the import-time HTTP call, see 5.2).
- `black` and `pre-commit` are currently declared as **runtime** dependencies in pyproject — move to a dev dependency group and actually enforce via pre-commit + CI.

### 2.2 Formatting & lint baseline
- `gas/cache/content_manager.py` is entirely **tab-indented** (811 lines) in an otherwise 4-space codebase. Reformat the repo once with ruff-format, commit as a standalone `style:` commit, add `.git-blame-ignore-revs`.
- Ruff rules to enable and burn down: `E722` (8 bare excepts), `F401` (unused imports), `T201` (174 `print()` calls in library code), `B` (bugbear).

### 2.3 Logging: one system
- `media_storage.py` (12 prints), `cache/util/video.py` (10), `miner_requests.py` (11), `util/image.py` — all log via `print()`; errors are invisible in pm2/Docker logs. Standardize on `bt.logging` (or a thin wrapper module so you can swap later).

### 2.4 Exception-handling policy
- 255 `except Exception` + 8 bare `except:` across the codebase, most log-and-continue, several `except: pass` (`generator_service.py:245,346,421,683,699,822`, `huggingface_uploads.py` ×4). Adopt a rule: catch narrow, log with stack trace, never `pass` silently; broad catches allowed only at loop/service top level. Burn down the worst offenders (the verification and DB paths first — that's where silent failure costs money).

### 2.5 Branch pruning
- 53 remote branches, many stale (`codex/*`, `feat/gs-v4`, `im/temp-burn-increase`...). Prune merged/dead ones.

---

## Phase 3 — Deduplication (delete ~2,000 lines without losing a feature)

Ordered by lines recovered per effort:

1. **CLI duplicate** — root `cli.py` deletion (Phase 0) recovers 1,129 lines. Also: within `gas/cli.py`, collapse the duplicated env-loader (`load_env`/`load_miner_env`), the three near-identical install commands, and the ANSI/time-ago helpers duplicated with `miner_stats.py`.
2. **Polling video-job services** — the submit→checkpoint→poll→timeout→download algorithm exists ~4× (`openai_service.py` main + resume, `openrouter_service.py` main + resume, `runway_service.py`). Extract a `PollingJobService` base with abstract `_submit()/_poll_status()/_download()`; resume-from-checkpoint becomes "enter the loop with an existing job id" instead of a copy of the whole method. ~350–450 lines recovered, and fixes the incoherent `process` vs `process_with_checkpoint` vs sync/async interface in `base_service.py` (currently the miner sniffs `iscoroutinefunction` to decide how to call a service).
3. **`miner_stats.py` parallel DB layer** — 1,005 lines re-implementing 11 raw SQL queries against tables owned by `gas/cache/db/` stores, spawned as a subprocess by the CLI. Guaranteed schema drift. Fold its queries into the stores and make it a thin presenter.
4. **Local-first model loader** — the `try local_files_only / except download` block is copy-pasted ~8× across `prompt_generator.py`, `util/model.py`, `generation_pipeline.py`. One helper taking a loader callable.
5. **C2PA manifest walking** — "resolve active manifest" copy-pasted 4× in `c2pa_verification.py`. One `_get_active_manifest()`.
6. **`data_service.py` substrate lifecycle** — hand-copied from `BaseNeuron` (`data_service.py:108-217` ≈ `base.py:58-181`). Fixed properly by Phase 4.1.
7. **Misc**: duplicated model-style config entries (`flux`≈`flux.1-dev`, `sdxl`≈`stable-diffusion-xl`, `cogvideo`≈`cogvideox`, `hunyuan`≈`hunyuanvideo`), triple-copied upload block in `push_model.py:123-216`, duplicated reward computation per modality in `rewards.py:165-183`, copy-pasted `source_filter` in `media_store.py`.

### Dead code to delete outright
- ✅ `gas/scraping/` (deleted 2026-07-08 — zero importers; took selenium+stamina deps, the Chrome apt blocks in `install.sh`/`Dockerfile`, and xvfb/libnss3 runtime packages with it; `source_type='scraper'` stays in the DB schema for historical rows)
- `local_service.py` (345 lines, disabled in registry, imports torch/diffusers at module top anyway)
- 8 of 13 `ModelPromptConfig` fields + the ~300 lines of `MODEL_STYLES` config that nothing reads (`model_prompt_styles.py`)
- CLIP "consensus" machinery built for a list of exactly one model (~130 lines, `clip_utils.py:415-544`)
- `generate_prompts_from_image`/`generate_scene_from_image` (zero callers), `is_duplicate` in `duplicate_detection.py` (exported, zero callers), `save_images_to_disk`
- Commented-out model configs in `models.py` (55 lines), commented-out `gen_tps`/`gen_local` job dispatch + everything it strands in `generator_service.py` (either re-enable behind a flag or delete the warm corpse)
- `get_service_info` methods never called by the registry; `__import__("sqlite3").Row` hack ×9 → one top-level import

---

## Phase 4 — Architecture (the real surgery)

### 4.1 Tell the truth about the process topology
`neurons/validator/services/` contains two things that are **not validator services**: `generator_service.py` and `data_service.py` are standalone processes with their own `main()`, own substrate connections, own config parsing. The actual system is **three processes racing on one shared SQLite database** (`ContentManager`) with no schema owner, no interface, no supervision — if `generator_service` dies, the validator keeps setting weights off stale cache data with no alarm.

- Move them out of `validator/services/` (e.g. `neurons/generator_service/`, `neurons/data_service/`) so the directory structure stops lying.
- Make all four long-running processes inherit one lifecycle base (`BaseNeuron`, fixed per Phase 1 #16): substrate connection, signal handling, clean SIGTERM shutdown. `data_service` currently copy-pastes all of this; the two standalone services currently only die cleanly on Ctrl-C.
- Define the cache handoff contract explicitly: the DB stores are the single access layer (no raw SQL from orchestration code — `generator_service.py:463-479` currently reaches through `ContentManager` into the connection), documented table ownership, and a heartbeat row per process so the validator can detect a dead sibling instead of silently zeroing scores 24h later.

### 4.2 Break up the god objects
Worst offenders, in order:
- **`GeneratorService` (871 lines, ~10 responsibilities)** → split into `VramProfiler`, `PromptPipeline`, `MediaGenerationRunner`, `VerificationRunner` with the service as a thin scheduler. `_generate_prompts` alone is 135 lines.
- **`ContentManager` (953 lines)** → it's a facade where half the methods are 1-line pass-throughs to the stores; either make callers use the stores directly or keep only the methods that add transactional value. Also the 120-line `upload_batch_to_huggingface`.
- **`Validator.set_weights` (100 lines)** → extract weight-budget computation (pure, testable) from extrinsic submission. Move the hardcoded budget (`burn=.2, video=.35, image=.3, audio=.0, generator=.15`), escrow addresses (currently a `HOTFIX` hardcode in the reward-critical path), EMA alpha, and inactivity cutoff into config.
- **`verify_media` (270 lines), `verify_c2pa` (140), `_generate_media_with_model` (160), `store_binary_content` (158), `generative_callback` (150)** → same treatment: extract pure decision logic from I/O.

### 4.3 One scheduler, one concurrency model
Right now scheduling is: block callbacks + a step-counter sleep loop (validator, two mechanisms in one class), nested `while True: time.sleep(60/30/10)` loops (generator service), `while True` daemon threads with no stop event (miner ×2, webhook saver), `asyncio.run()` per task inside worker threads (miner, plus unbounded thread-per-restored-task fan-out that bypasses the concurrency cap), and a heartbeat thread that recovers from hangs by calling `sys.exit(0)` from a non-main thread.

- Pick asyncio as the backbone per process. Long-running work: tasks with cancellation, not daemon threads. Sync SDK calls: `asyncio.to_thread`.
- One periodic-job scheduler abstraction (interval + block-interval jobs) replacing the 38 `time.sleep()`s used as scheduling.
- Proper shutdown: every loop gets a stop event; SIGTERM drains in-flight generations (the miner currently orphans them); no `sys.exit` outside `main()` (also `base.py:50` calling `exit()` from a library method, and `push_model.py`'s infinite `while True` retry with no max attempts).

### 4.4 Config: one schema, no env soup
Config currently lives in: argparse (~60 args, with `--device` defined twice with *different* defaults, `--prompt-modalities` defined twice, `callback_port` vs `callback-port`), direct `os.getenv` reads scattered through services, `getattr(config, x, default)` probing with defaults duplicated across call sites, hardcoded constants (`gas.bitmind.ai` in 4+ places, stake 20000 twice, `MAINNET_UID`), and two pm2 `.js` files.

- Introduce a typed settings object (pydantic-settings fits: env + CLI + defaults in one schema, validated at startup). Constructors receive settings; nothing reads `os.getenv` below `main()`.
- All reward/policy knobs (weight budget, EMA alpha, fool-rate bonus params, thresholds) become config with defaults — changing incentive policy should not require a code edit.

### 4.5 Database layer
- Fix migrations: currently a failed migration is logged and the runner **continues to the next one** — silent schema drift is why `_row_to_media_entry` has defensive `"col" in row.keys()` checks. Fail hard, run each migration in a transaction, track applied migrations in a table.
- Kill the string-encoded hash protocol in `duplicate_detection.py` (`"phash1,phash2|seg1;seg2_framecount"` parsed by `rsplit`) — store structured JSON.
- Fix the N+1 in `get_unuploaded_media` (per-row prompt lookup → JOIN).

---

## Phase 5 — Performance

1. **Cache loaded diffusion models.** `generation_pipeline.py` does `clear_gpu()` + full `from_pretrained` on **every generation**, then `del self.model` at the end. Same model used for N prompts = N full loads from disk (tens of seconds each). Add an LRU keyed by model name, evict on VRAM pressure. Biggest single win in the repo.
2. **Stop decoding media three times per submission.** Each callback writes the bytes to a temp file and fully decodes for corruption-check, again for perceptual hash, again for C2PA. Write once, decode once, pass frames/path through the pipeline.
3. **`sample_frames` spawns one ffmpeg subprocess per frame** (up to 24, inside a 5-retry loop → up to 120 process spawns per clip). One ffmpeg invocation with a `select`/`fps` filter.
4. **CLIP video scoring is unbatched** — per-video Python loop, re-opening the file and seeking frame-by-frame with `cap.set(POS_FRAMES)`. Batch frames across videos; read sequentially.
5. **No import-time side effects**: `rewards.py` makes a blocking HTTP call to OpenRouter at module import; `models.py` creates CUDA `torch.Generator`s at import (shared RNG state across all generations); `GenerativeChallengeManager.__init__` and `miner.py` construction both block on `checkip.amazonaws.com`. Make all of these lazy — the CI import-smoke test from 2.1 then guards this forever.
7. **In-batch duplicate check is O(n²)** (`verification_pipeline.py:309-315`) and DB dup-check is a linear scan of 1000 rows per submission; fine for now, but the docstring already says "consider LSH/FAISS" — do that when volume warrants.

---

## Phase 6 — Testing (from 2.6% to trustworthy)

Current state: 737 test lines / 39 tests against 28k LOC, all in the prompt/verification corner. No tests for: validator scoring/weights, rewards math, cache DB layer, CLI, autoupdater, protocol, scraping. `tests/generator/stabilityai_service.py` is missing the `test_` prefix so pytest never collects it.

Priority order (test the money first):
1. **Rewards & weight budget math** — pure functions after the Phase 4.2 extraction; property tests (weights sum to 1, no NaN propagation, EMA bounds).
2. **DB stores** — in-memory SQLite; would have caught the wrong-table DELETE, the SQL interpolation, and the N+1. Migration tests: fresh DB vs migrated DB produce identical schemas.
3. **Verification decisions** — C2PA trust matching (exact-match table tests, including the adversarial "issuer contains 'Google'" case), dup-detection hash round-trip, threshold behavior.
4. **`PollingJobService`** — one mocked-HTTP test suite covering submit/poll/timeout/resume for all providers at once (payoff of Phase 3 #2).
5. **Epistula signature verification** — valid/stale/future/tampered.
6. **A smoke test per process entrypoint** — construct with fake config, run one loop iteration, shut down cleanly. Forces the testability fixes (no `exit()` in constructors, no network in `__init__`, no class-level state) to stick.

---

## Suggested sequencing

| When | What | Outcome |
|------|------|---------|
| Today | Phase 0 | Stray env file gone, tree clean, gitignore fixed |
| Week 1 | Phase 1 + 2.1–2.2 | 20 bugs fixed, CI + format baseline live |
| Week 2 | 2.3–2.5, Phase 3 items 1–4 | One logging system, ~2k lines deleted |
| Weeks 3–4 | Phase 4.1, 4.4, 4.5 | Honest process topology, typed config, safe migrations |
| Weeks 5–6 | Phase 4.2, 4.3 + Phase 6 items 1–3 | God objects split, one scheduler, money-path tested |
| Ongoing | Phase 5, Phase 6 items 4–6 | Model cache (do early — it's cheap and huge), the rest opportunistically |

Rule of thumb for the whole effort: **every Phase 4 extraction lands with its Phase 6 test.** Refactoring this codebase without tests is how it got here.
