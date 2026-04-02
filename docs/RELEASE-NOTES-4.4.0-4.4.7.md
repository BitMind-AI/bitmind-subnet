# Release notes: 4.4.0 – 4.4.7

Summary of changes for BitMind subnet (GAS / SN34) releases from 4.4.0 through 4.4.7.

---

## 4.4.0 (31 Jan 2026)

### Summary

- Remove LocalService from registry (no C2PA-signed content)
- Add C2PA verification CLI for testing C2PA credentials on local files
- Add webhook stats tracking for miners (success/failure per validator)
- Add detailed error responses for generative callback failures
- Add prompt cache cleanup to avoid unbounded growth
- Add prompt modality tracking (image/video/audio)
- Dynamic escrow address fetching from API with fallback
- Fix file format handling (magic bytes instead of hardcoded extensions/content-types)
- Improve media format detection (ftyp-based MP4, extra formats)
- Fix generator reward logic for sharper penalties when miners stop responding

### Changes

**Generator miner rewards**

- Hard cutoff to zero out scores for generators inactive in the liveness window
- EMA alpha increased from `0.2` to `0.5` for faster score decay when inactive
- Base reward default `1e-4` → `0` for generators without verified submissions
- Reward eligibility: union → intersection (need both local verified submissions and benchmark results)
- Fix: `include_all=True` no longer double-counts already-rewarded media in verification stats

**Media format detection (#311)**

- Better JPEG detection (3-byte signature), length checks
- Audio: MP3, WAV, FLAC, OGG, M4A
- Video: MOV, 3GP, AVI
- Image: HEIC/HEIF, AVIF, GIF, BMP, TIFF
- ftyp-based detection consolidated; MP4 fixed (ftyp at offset 4)

**Dynamic escrow addresses (#322)**

- Escrow eligibility based on validators running latest subnet code
- Validators fetch `video_escrow`, `image_escrow`, `audio_escrow` from gas-api
- New `get_escrow_addresses()` for optional weight-copier punishment

**Prompt modality**

- Challenge manager chooses modality first, then samples matching prompts
- Prompts stored with intended modality; `modality` column in prompts table (`image` / `video` / `audio`)
- Migration adds column and fixes ordering to avoid “no such column: modality”

**Prompt cache cleanup**

- Prompts deleted after use when enough remain
- `min_prompts_threshold` (default 100) to avoid running out
- Fix: `remove` in prompt sampling now deletes prompts instead of only incrementing `used_count`

**Generative callback error responses**

- `400 - C2PA verification failed: untrusted issuer`
- `400 - C2PA verification failed: no C2PA manifest`
- `400 - Duplicate content detected`
- `400 - Corrupted or unreadable media`
- `400 - Empty binary payload`

**Miner webhook stats**

- Auto-cleanup of archives older than 7 days; daily rotation to `webhook_stats_archive/webhook_stats_YYYY-MM-DD.json`
- Summary in logs every 5 minutes; debounced save every 60 seconds
- Failure categories: `http_400`, `http_401`, `http_404`, `http_500`, `connection_error`, `connection_timeout`, `empty_payload`
- Helpers: `reset_webhook_stats()`, `get_webhook_stats_json()`, `print_webhook_stats()`

**LocalService removed from registry**

- Valid services: `openai`, `openrouter`, `stabilityai`, or `none`
- LocalService remains in codebase but not in `SERVICE_MAP` (no C2PA)

**C2PA verification CLI**

- `gascli generator verify-c2pa` (and `neurons/generator/helper/verify_c2pa.py`)
- Options: `--verbose`, `--json`

---

## 4.4.1 (30 Jan 2026)

- **Generator prompt sampling:** Use `remove=False` when sampling prompts so prompts are not deleted after use (avoids exhausting the prompt pool).
- **Validator weight normalization:** Only assign non-zero weights to UIDs in the active set; explicitly zero out weights for inactive generative miners before scaling, so inactive miners no longer receive weight.

---

## 4.4.2 (31 Jan 2026)

- **C2PA verification:** Broader C2PA issuer handling so more certificate issuers are accepted as trusted.
- Allow all certificate issuers to be treated as CA issuers where appropriate.
- Remove redundant issuer logic and consolidate issuer handling.
- Increase C2PA options and clean up multiple issuer versions.

---

## 4.4.3 (3 Feb 2026)

- **New upload endpoint** for discriminative miner model uploads (`neurons/discriminator/push_model.py`).
- **Validator:** Decrease burn (burn rate reduced).

---

## 4.4.4 (4 Feb 2026)

- **Generator prompt sampling:** Keep `remove=False` so prompts are not removed when sampled; use `threading.Lock` for thread-safe prompt sampling.
- **Content cache and uploads:** New content DB and content manager support plus a new upload path used by the data service.
- **Config:** New options for content/upload behavior (see `.env.validator.template`).
- **Docs:** Updates to Discriminative-Mining, ONNX, and Validating.

---

## 4.4.5 (6 Feb 2026)

- **Generator rewards:** Use a **2-hour lookback window** for recent verified miner media when computing generator base rewards (eligibility and score stability).
- **Prompt server:** New prompt server implementation (`gas/evaluation/prompt_server.py`) for serving challenges to miners.
- **Validator:** Uses new prompt server and reward lookback; removal of obsolete validator-request code.
- **Content cache:** Content DB and content manager extended to support the new lookback and verification stats.

---

## 4.4.6 (6 Feb 2026)

- **Stability AI service:** Fix request headers for the Stability AI miner service so API calls succeed.
- **Webhooks / logging:** Remove unused `has_c2pa` key from logging; keep useful logging; reduce log noise.
- **Metagraph / validator:** Minor adjustments for consistency and clarity.

---

## 4.4.7 (8 Feb 2026)

- **Verification stats:** Add support for **failed** miner media in the lookback window.
  - New `get_recent_failed_miner_media()` in content DB and content manager (default 2-hour lookback).
  - New `get_verification_stats_last_n_hours()` to compute pass/fail counts and pass rate per miner over the last N hours (for generator base rewards and eligibility).
- **Content manager:** New helper `_build_verification_stats_from_verified_and_failed()` to build per-miner verification stats from both verified and failed media.
- **Validator:** Uses the new verification stats and failed-media lookback for more accurate generator rewards and eligibility.

---

*Generated from repository history and release merge messages.*
