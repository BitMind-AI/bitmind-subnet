#!/usr/bin/env python3
"""
Miner verification & challenge statistics.

Usage:
    gascli v miner-stats                              # summary table of all miners
    gascli v miner-stats --by-coldkey                 # group by coldkey via metagraph
    gascli v miner-stats --uid 5                      # detailed breakdown for UID 5
    gascli v miner-stats --uid 5 --limit 50            # last 50 challenge results
    gascli v miner-stats --by-coldkey --lookback-hours 48
"""

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError, TypeError):
        return str(ts)


def fmt_ago(ts: float | None) -> str:
    """Human-friendly relative time."""
    if ts is None:
        return ""
    delta = time.time() - ts
    if delta < 0:
        return "just now"
    mins = int(delta // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins}m ago"
    hrs = mins // 60
    if hrs < 24:
        return f"{hrs}h {mins % 60}m ago"
    days = hrs // 24
    return f"{days}d {hrs % 24}h ago"


def fmt_hotkey(hk: str | None) -> str:
    if not hk:
        return "N/A"
    if len(hk) > 16:
        return f"{hk[:8]}...{hk[-8:]}"
    return hk


def fmt_coldkey(ck: str | None) -> str:
    """Truncate coldkey for display."""
    if not ck:
        return "N/A"
    if len(ck) > 16:
        return f"{ck[:8]}...{ck[-8:]}"
    return ck


def pct(part: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{part / total * 100:.1f}%"


def bar(value: float, width: int = 20) -> str:
    """Render a tiny coloured bar for pass rates."""
    filled = int(round(value * width))
    empty = width - filled
    if value >= 0.8:
        colour = Colors.GREEN
    elif value >= 0.5:
        colour = Colors.YELLOW
    else:
        colour = Colors.RED
    return f"{colour}{'█' * filled}{Colors.DIM}{'░' * empty}{Colors.END}"


def yes_no(val: bool | int | None) -> str:
    if val:
        return f"{Colors.GREEN}✅ Yes{Colors.END}"
    return f"{Colors.RED}❌ No{Colors.END}"


def status_badge(status: str) -> str:
    badges = {
        "verified": f"{Colors.GREEN}✅ VERIFIED{Colors.END}",
        "failed": f"{Colors.RED}❌ FAILED{Colors.END}",
        "pending": f"{Colors.YELLOW}⏳ PENDING{Colors.END}",
        "stored": f"{Colors.CYAN}💾 STORED{Colors.END}",
    }
    return badges.get(status, f"{Colors.DIM}{status.upper()}{Colors.END}")


# Column layout shared between flat and grouped views
COLS = [
    ("UID",      5),
    ("Hotkey",  18),
    ("Total",    6),
    ("Passed",   7),
    ("Failed",   7),
    ("Pend",     5),
    ("Pass%",    7),
    ("⬆️ Up",     5),
    ("⬇️ NoUp",   6),
    ("Rewarded", 9),
    ("Bar",     22),
    ("Last Seen", 17),
]

COLS_WITH_COLDKEY = [
    ("UID",      5),
    ("Hotkey",  18),
    ("Coldkey", 18),
    ("Total",    6),
    ("Passed",   7),
    ("Failed",   7),
    ("Pend",     5),
    ("Pass%",    7),
    ("⬆️ Up",     5),
    ("⬇️ NoUp",   6),
    ("Rewarded", 9),
    ("Bar",     22),
    ("Last Seen", 17),
]

TOTAL_WIDTH = sum(w for _, w in COLS) + len(COLS) - 1
TOTAL_WIDTH_COLDKEY = sum(w for _, w in COLS_WITH_COLDKEY) + len(COLS_WITH_COLDKEY) - 1


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def _get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def query_miner_summaries(
    db_path: Path,
    lookback_hours: float | None = None,
) -> list[dict]:
    """Return one row per miner with aggregate stats."""
    conn = _get_conn(db_path)
    try:
        where = "WHERE source_type = 'miner'"
        params: tuple = ()
        if lookback_hours is not None:
            cutoff = time.time() - lookback_hours * 3600
            where += " AND created_at >= ?"
            params = (cutoff,)

        sql = f"""
            SELECT
                uid,
                hotkey,
                COUNT(*)                                                     AS total_media,
                SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END)                AS verified,
                SUM(CASE WHEN failed_verification = 1 THEN 1 ELSE 0 END)     AS failed,
                SUM(CASE WHEN verified = 0 AND failed_verification = 0 THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN uploaded = 1 THEN 1 ELSE 0 END)                AS uploaded,
                SUM(CASE WHEN uploaded = 0 OR uploaded IS NULL THEN 1 ELSE 0 END) AS not_uploaded,
                SUM(CASE WHEN rewarded = 1 THEN 1 ELSE 0 END)                AS rewarded,
                MAX(created_at)                                              AS last_activity
            FROM media
            {where}
            GROUP BY uid, hotkey
            ORDER BY uid ASC
        """
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def query_challenge_outcomes(
    db_path: Path,
    uid: int,
    limit: int = 20,
    lookback_hours: float | None = None,
) -> list[dict]:
    """Return recent challenge outcomes for a specific miner."""
    conn = _get_conn(db_path)
    try:
        where = "WHERE uid = ?"
        params: list = [uid]
        if lookback_hours is not None:
            cutoff = time.time() - lookback_hours * 3600
            where += " AND updated_at >= ?"
            params.append(cutoff)

        sql = f"""
            SELECT task_id, uid, hotkey, prompt_id, modality, status,
                   failure_reason, media_id, created_at, updated_at
            FROM generator_challenge_outcomes
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def query_miner_media(
    db_path: Path,
    uid: int,
    limit: int = 20,
    lookback_hours: float | None = None,
) -> list[dict]:
    """Return recent media entries for a specific miner."""
    conn = _get_conn(db_path)
    try:
        where = "WHERE source_type = 'miner' AND uid = ?"
        params: list = [uid]
        if lookback_hours is not None:
            cutoff = time.time() - lookback_hours * 3600
            where += " AND created_at >= ?"
            params.append(cutoff)

        sql = f"""
            SELECT id, uid, hotkey, file_path, modality, media_type,
                   verified, failed_verification, uploaded, rewarded,
                   task_id, model_name, c2pa_verified, c2pa_issuer,
                   perceptual_hash, created_at, file_size, format
            FROM media
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def query_grand_totals(db_path: Path) -> dict:
    """Overall network-wide miner stats."""
    conn = _get_conn(db_path)
    try:
        row = conn.execute("""
            SELECT
                COUNT(DISTINCT uid)                        AS total_miners,
                COUNT(*)                                   AS total_media,
                SUM(CASE WHEN verified = 1 THEN 1 ELSE 0 END)     AS total_verified,
                SUM(CASE WHEN failed_verification = 1 THEN 1 ELSE 0 END) AS total_failed,
                SUM(CASE WHEN uploaded = 1 THEN 1 ELSE 0 END)       AS total_uploaded,
                SUM(CASE WHEN rewarded = 1 THEN 1 ELSE 0 END)       AS total_rewarded
            FROM media
            WHERE source_type = 'miner'
        """).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Metagraph coldkey lookup
# ---------------------------------------------------------------------------
def resolve_coldkeys(
    rows: list[dict],
    netuid: int,
    chain_endpoint: str | None = None,
) -> dict[int, str]:
    """Connect to subtensor and return uid→coldkey mapping for UIDs in *rows*.

    Uses BT_CHAIN_ENDPOINT env var if --chain-endpoint not provided,
    falls back to finney network.
    """
    import os as _os

    try:
        import bittensor as bt

        # Resolve endpoint: explicit arg > env var > finney network
        endpoint = (
            chain_endpoint
            or _os.environ.get("BT_CHAIN_ENDPOINT")
            or _os.environ.get("BT_SUBTENSOR_CHAIN_ENDPOINT")
        )

        if endpoint:
            print(f"  {Colors.DIM}connecting to subtensor: {endpoint}{Colors.END}")
            subtensor = bt.subtensor(chain_endpoint=endpoint)
        else:
            print(f"  {Colors.DIM}connecting to subtensor: finney{Colors.END}")
            subtensor = bt.subtensor(network="finney")

        print(f"  {Colors.DIM}fetching metagraph for netuid {netuid}...{Colors.END}")
        metagraph = subtensor.metagraph(netuid=netuid, lite=True)
        subtensor.close()

        n_neurons = int(metagraph.n.item())
        coldkeys_list = metagraph.coldkeys

        mapping: dict[int, str] = {}
        for row in rows:
            uid = row["uid"]
            if uid is not None and uid < n_neurons:
                ck = coldkeys_list[uid]
                if ck:
                    mapping[uid] = ck

        print(f"  {Colors.DIM}resolved {len(mapping)} coldkeys{Colors.END}")
        return mapping

    except Exception as e:
        print(f"\n  {Colors.YELLOW}⚠️  Could not resolve coldkeys from subtensor:{Colors.END}")
        print(f"  {Colors.DIM}{e}{Colors.END}")
        return {}


# ---------------------------------------------------------------------------
# Grouped view (by coldkey)
# ---------------------------------------------------------------------------
def _build_group_coldkey_header(
    coldkey: str,
    group_total_media: int,
    group_verified: int,
    group_failed: int,
    group_rewarded: int,
) -> None:
    """Print a coldkey group header with aggregate stats for all UIDs in that group."""
    evaluated = group_verified + group_failed
    rate = group_verified / max(evaluated, 1)
    print()
    print(
        f"  {Colors.BOLD}{Colors.CYAN}┌─ Coldkey:{Colors.END} "
        f"{fmt_coldkey(coldkey)}  "
        f"│  {Colors.DIM}UIDs in group:{Colors.END} (see below)  "
        f"│  med: {group_total_media}  "
        f"pass: {group_verified}  fail: {group_failed}  "
        f"rate: {pct(group_verified, evaluated):>6}  "
        f"{bar(rate, 12)}  "
        f"rewarded: {group_rewarded}"
    )


def print_miner_summary_grouped(
    rows: list[dict],
    coldkey_map: dict[int, str],
    lookback_hours: float | None = None,
) -> None:
    """Print the miner summary table grouped by coldkey."""
    if not rows:
        print(f"  {Colors.YELLOW}No miner media found in the database.{Colors.END}")
        return

    # Group rows by coldkey, sorted by (coldkey, uid)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        uid = r["uid"]
        ck = coldkey_map.get(uid) or f"unknown-{uid}"
        groups[ck].append(r)

    # Sort groups: known coldkeys first (alphabetically), unknown last
    def sort_key(item: tuple[str, list[dict]]) -> tuple[int, str]:
        ck, _ = item
        is_unknown = ck.startswith("unknown-")
        return (int(is_unknown), ck)

    sorted_groups = sorted(groups.items(), key=sort_key)

    # Header
    cols = COLS_WITH_COLDKEY
    header = " ".join(f"{Colors.BOLD}{name:<{w}}{Colors.END}" for name, w in cols)
    print(f"\n{Colors.BOLD}{Colors.HEADER}📊 Miner Verification Summary (grouped by coldkey){Colors.END}")
    if lookback_hours:
        print(f"   {Colors.DIM}lookback: last {lookback_hours}h{Colors.END}")
    print()
    print(header)
    print("─" * TOTAL_WIDTH_COLDKEY)

    for coldkey, group_rows in sorted_groups:
        # Compute group aggregates
        group_total = sum(r["total_media"] or 0 for r in group_rows)
        group_verified = sum(r["verified"] or 0 for r in group_rows)
        group_failed = sum(r["failed"] or 0 for r in group_rows)
        group_rewarded = sum(r["rewarded"] or 0 for r in group_rows)

        _build_group_coldkey_header(
            coldkey, group_total, group_verified, group_failed, group_rewarded
        )

        # Print each row in the group (with coldkey column blank since it's in header)
        for r in group_rows:
            total = r["total_media"] or 0
            passed = r["verified"] or 0
            failed = r["failed"] or 0
            pending = r["pending"] or 0
            uploaded = r["uploaded"] or 0
            not_up = r["not_uploaded"] or 0
            rewarded = r["rewarded"] or 0
            pass_rate = passed / max(passed + failed, 1)
            uid_str = str(r["uid"]) if r["uid"] is not None else "?"
            hk = fmt_hotkey(r["hotkey"])
            ck = fmt_coldkey(coldkey)
            ago = fmt_ago(r["last_activity"])

            row_str = (
                f"  {Colors.DIM}├─{Colors.END} "
                f"{uid_str:<5} "
                f"{hk:<18} "
                f"{Colors.DIM}{ck:<18}{Colors.END} "
                f"{total:<6} "
                f"{passed:<7} "
                f"{failed:<7} "
                f"{pending:<5} "
                f"{pct(passed, passed + failed):<7} "
                f"{uploaded:<5} "
                f"{not_up:<6} "
                f"{rewarded:<9} "
                f"{bar(pass_rate):<22} "
                f"{ago:<17}"
            )
            print(row_str)

        # Close group
        n_uids = len(group_rows)
        evaluated = group_verified + group_failed
        print(
            f"  {Colors.DIM}└─ {n_uids} UID(s) in group{Colors.END}"
        )

    print("─" * TOTAL_WIDTH_COLDKEY)


# ---------------------------------------------------------------------------
# Flat view (original)
# ---------------------------------------------------------------------------
def print_miner_summary_table(
    rows: list[dict],
    lookback_hours: float | None = None,
) -> None:
    """Print the all-miners summary table (flat, sorted by UID)."""
    if not rows:
        print(f"  {Colors.YELLOW}No miner media found in the database.{Colors.END}")
        return

    # Header
    header = " ".join(f"{Colors.BOLD}{name:<{w}}{Colors.END}" for name, w in COLS)
    print(f"\n{Colors.BOLD}{Colors.HEADER}📊 Miner Verification Summary{Colors.END}")
    if lookback_hours:
        print(f"   {Colors.DIM}lookback: last {lookback_hours}h{Colors.END}")
    print()
    print(header)
    print("─" * TOTAL_WIDTH)

    for r in rows:
        total = r["total_media"] or 0
        passed = r["verified"] or 0
        failed = r["failed"] or 0
        pending = r["pending"] or 0
        uploaded = r["uploaded"] or 0
        not_up = r["not_uploaded"] or 0
        rewarded = r["rewarded"] or 0
        pass_rate = passed / max(passed + failed, 1)
        uid_str = str(r["uid"]) if r["uid"] is not None else "?"
        hk = fmt_hotkey(r["hotkey"])
        ago = fmt_ago(r["last_activity"])

        row_str = (
            f"{uid_str:<5} "
            f"{hk:<18} "
            f"{total:<6} "
            f"{passed:<7} "
            f"{failed:<7} "
            f"{pending:<5} "
            f"{pct(passed, passed + failed):<7} "
            f"{uploaded:<5} "
            f"{not_up:<6} "
            f"{rewarded:<9} "
            f"{bar(pass_rate):<22} "
            f"{ago:<17}"
        )
        print(row_str)

    print("─" * TOTAL_WIDTH)


def query_multi_uid_challenges(
    db_path: Path,
    uids: list[int],
    limit: int = 50,
    lookback_hours: float | None = None,
) -> list[dict]:
    """Return recent challenge outcomes for multiple UIDs."""
    if not uids:
        return []
    conn = _get_conn(db_path)
    try:
        placeholders = ",".join("?" * len(uids))
        where = f"WHERE uid IN ({placeholders})"
        params: list = list(uids)
        if lookback_hours is not None:
            cutoff = time.time() - lookback_hours * 3600
            where += " AND updated_at >= ?"
            params.append(cutoff)

        sql = f"""
            SELECT task_id, uid, hotkey, prompt_id, modality, status,
                   failure_reason, media_id, created_at, updated_at
            FROM generator_challenge_outcomes
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def query_multi_uid_media(
    db_path: Path,
    uids: list[int],
    limit: int = 50,
    lookback_hours: float | None = None,
) -> list[dict]:
    """Return recent media entries for multiple UIDs."""
    if not uids:
        return []
    conn = _get_conn(db_path)
    try:
        placeholders = ",".join("?" * len(uids))
        where = f"WHERE source_type = 'miner' AND uid IN ({placeholders})"
        params: list = list(uids)
        if lookback_hours is not None:
            cutoff = time.time() - lookback_hours * 3600
            where += " AND created_at >= ?"
            params.append(cutoff)

        sql = f"""
            SELECT id, uid, hotkey, file_path, modality, media_type,
                   verified, failed_verification, uploaded, rewarded,
                   task_id, model_name, c2pa_verified, c2pa_issuer,
                   perceptual_hash, created_at, file_size, format
            FROM media
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def print_grand_totals(totals: dict) -> None:
    """Print network-wide aggregate line."""
    if not totals or not totals.get("total_media"):
        return
    t = totals
    print()
    print(f"  {Colors.DIM}Network totals — "
          f"miners: {t['total_miners']}  "
          f"media: {t['total_media']}  "
          f"verified: {t['total_verified']}  "
          f"failed: {t['total_failed']}  "
          f"uploaded: {t['total_uploaded']}  "
          f"rewarded: {t['total_rewarded']}{Colors.END}")


def print_miner_detail(
    uid: int,
    db_path: Path,
    limit: int,
    lookback_hours: float | None,
    coldkey_map: dict[int, str] | None = None,
) -> None:
    """Print a detailed view for a single miner."""

    # 1. Summary card
    rows = query_miner_summaries(db_path, lookback_hours)
    miner_row = next((r for r in rows if r["uid"] == uid), None)
    if miner_row is None:
        print(f"\n  {Colors.RED}❌ No miner media found for UID {uid}{Colors.END}")
        return

    r = miner_row
    total = r["total_media"] or 0
    passed = r["verified"] or 0
    failed = r["failed"] or 0
    pending = r["pending"] or 0
    uploaded = r["uploaded"] or 0
    not_up = r["not_uploaded"] or 0
    rewarded = r["rewarded"] or 0
    evaluated = passed + failed
    pass_rate = passed / max(evaluated, 1)
    hk = r["hotkey"] or "unknown"
    ck = coldkey_map.get(uid) if coldkey_map else None

    print(f"\n{Colors.BOLD}{Colors.HEADER}🔍 Miner UID {uid} — Detailed Report{Colors.END}")
    if lookback_hours:
        print(f"   {Colors.DIM}lookback: last {lookback_hours}h{Colors.END}")
    print(f"\n  {Colors.BOLD}Hotkey:{Colors.END}  {hk}")
    if ck:
        print(f"  {Colors.BOLD}Coldkey:{Colors.END} {ck}")
    print(f"  {Colors.BOLD}Last Activity:{Colors.END} {fmt_ts(r['last_activity'])} ({fmt_ago(r['last_activity'])})")

    print(f"\n  {Colors.CYAN}── Verification ──{Colors.END}")
    print(f"  Total Media:       {total}")
    print(f"  Evaluated:         {evaluated}  (pass: {passed}  fail: {failed}  pending: {pending})")
    print(f"  Pass Rate:         {pct(passed, evaluated):>6}  {bar(pass_rate)}")
    print(f"  Uploaded / Not:    {uploaded} / {not_up}")
    print(f"  Rewarded:          {rewarded}")

    # 2. Recent challenge outcomes
    outcomes = query_challenge_outcomes(db_path, uid, limit, lookback_hours)
    print(f"\n  {Colors.CYAN}── Recent Challenge Outcomes ({len(outcomes)}) ──{Colors.END}")
    if not outcomes:
        print(f"  {Colors.DIM}(no challenge outcomes found){Colors.END}")
    else:
        # Header
        print(f"  {Colors.BOLD}{'Updated':<19} {'Status':<20} {'Modality':<9} {'Failure Reason':<45}{Colors.END}")
        print(f"  {'─' * 19} {'─' * 20} {'─' * 9} {'─' * 45}")
        for o in outcomes:
            reason = (o["failure_reason"] or "") if o["status"] == "failed" else ""
            if reason and len(reason) > 43:
                reason = reason[:40] + "..."
            status = status_badge(o["status"])
            mod = (o["modality"] or "")[:8]
            ts = fmt_ts(o["updated_at"])
            print(f"  {ts:<19} {status:<30} {mod:<9} {Colors.DIM if not reason else Colors.RED}{reason or '—':<45}{Colors.END}")

    # 3. Recent media entries
    media = query_miner_media(db_path, uid, limit, lookback_hours)
    print(f"\n  {Colors.CYAN}── Recent Media Entries ({len(media)}) ──{Colors.END}")
    if not media:
        print(f"  {Colors.DIM}(no media entries found){Colors.END}")
    else:
        for m in media:
            fname = Path(m["file_path"]).name if m["file_path"] else "?"
            verified = yes_no(m["verified"])
            uploaded = yes_no(m["uploaded"])
            rewarded = yes_no(m["rewarded"])
            mod = m["modality"] or "?"
            c2pa = yes_no(m["c2pa_verified"])
            ts = fmt_ts(m["created_at"])
            ago = fmt_ago(m["created_at"])

            print(f"  ┌─ {Colors.BOLD}{fname}{Colors.END}")
            print(f"  │  {Colors.DIM}{ts} ({ago}){Colors.END}")
            print(f"  │  Modality: {mod}  │  Verified: {verified}  │  Uploaded: {uploaded}  │  Rewarded: {rewarded}  │  C2PA: {c2pa}")
            if m.get("c2pa_issuer"):
                print(f"  │  C2PA Issuer: {m['c2pa_issuer']}")
            if m.get("task_id"):
                print(f"  │  Task ID: {m['task_id'][:16]}...")
            if m.get("file_size"):
                sz = m["file_size"]
                if sz < 1024:
                    sz_str = f"{sz}B"
                elif sz < 1024 * 1024:
                    sz_str = f"{sz / 1024:.1f}KB"
                else:
                    sz_str = f"{sz / (1024 * 1024):.1f}MB"
                print(f"  │  Size: {sz_str}  │  Format: {m.get('format', '?')}")
            print(f"  └─")


def print_coldkey_detail(
    coldkey: str,
    uids: list[int],
    rows: list[dict],
    db_path: Path,
    limit: int,
    lookback_hours: float | None,
) -> None:
    """Print a detailed view for all miners under one coldkey."""
    # Filter summary rows to just these UIDs
    group_rows = [r for r in rows if r["uid"] in uids]
    if not group_rows:
        print(f"\n  {Colors.RED}❌ No miner media found for coldkey{Colors.END}")
        return

    # Compute group aggregates
    group_total = sum(r["total_media"] or 0 for r in group_rows)
    group_verified = sum(r["verified"] or 0 for r in group_rows)
    group_failed = sum(r["failed"] or 0 for r in group_rows)
    group_pending = sum(r["pending"] or 0 for r in group_rows)
    group_uploaded = sum(r["uploaded"] or 0 for r in group_rows)
    group_rewarded = sum(r["rewarded"] or 0 for r in group_rows)
    evaluated = group_verified + group_failed
    pass_rate = group_verified / max(evaluated, 1)

    print(f"\n{Colors.BOLD}{Colors.HEADER}🧊 Coldkey: {coldkey}{Colors.END}")
    if lookback_hours:
        print(f"   {Colors.DIM}lookback: last {lookback_hours}h{Colors.END}")
    print(f"\n  {Colors.BOLD}UIDs:{Colors.END} {', '.join(str(u) for u in sorted(uids))}")

    print(f"\n  {Colors.CYAN}── Aggregate Stats ──{Colors.END}")
    print(f"  Total Media:       {group_total}")
    print(f"  Evaluated:         {evaluated}  (pass: {group_verified}  fail: {group_failed}  pending: {group_pending})")
    print(f"  Pass Rate:         {pct(group_verified, evaluated):>6}  {bar(pass_rate)}")
    print(f"  Uploaded / Not:    {group_uploaded} / {group_total - group_uploaded}")
    print(f"  Rewarded:          {group_rewarded}")

    # Per-UID summary table
    print(f"\n  {Colors.CYAN}── Per-UID Breakdown ──{Colors.END}")
    cols_small = [
        ("UID", 5), ("Hotkey", 18), ("Total", 6), ("Passed", 7), ("Failed", 7),
        ("Pass%", 7), ("⬆️ Up", 5), ("Rewarded", 9), ("Bar", 22), ("Last Seen", 17),
    ]
    header = " ".join(f"{Colors.BOLD}{n:<{w}}{Colors.END}" for n, w in cols_small)
    tw = sum(w for _, w in cols_small) + len(cols_small) - 1
    print(f"  {header}")
    print(f"  {'─' * tw}")
    for r in sorted(group_rows, key=lambda r: r["uid"] or 0):
        t = r["total_media"] or 0
        p = r["verified"] or 0
        f = r["failed"] or 0
        up = r["uploaded"] or 0
        rw = r["rewarded"] or 0
        pr = p / max(p + f, 1)
        print(
            f"  {str(r['uid']):<5} "
            f"{fmt_hotkey(r['hotkey']):<18} "
            f"{t:<6} {p:<7} {f:<7} "
            f"{pct(p, p+f):<7} {up:<5} {rw:<9} "
            f"{bar(pr):<22} {fmt_ago(r['last_activity']):<17}"
        )
    print(f"  {'─' * tw}")

    # Recent challenge outcomes across all UIDs
    outcomes = query_multi_uid_challenges(db_path, uids, limit, lookback_hours)
    print(f"\n  {Colors.CYAN}── Recent Challenge Outcomes ({len(outcomes)}) ──{Colors.END}")
    if not outcomes:
        print(f"  {Colors.DIM}(no challenge outcomes found){Colors.END}")
    else:
        print(f"  {Colors.BOLD}{'UID':<5} {'Updated':<19} {'Status':<20} {'Modality':<9} {'Failure Reason':<40}{Colors.END}")
        print(f"  {'─' * 5} {'─' * 19} {'─' * 20} {'─' * 9} {'─' * 40}")
        for o in outcomes:
            reason = (o["failure_reason"] or "") if o["status"] == "failed" else ""
            if reason and len(reason) > 38:
                reason = reason[:35] + "..."
            status = status_badge(o["status"])
            mod = (o["modality"] or "")[:8]
            ts = fmt_ts(o["updated_at"])
            print(
                f"  {str(o['uid']):<5} {ts:<19} {status:<30} {mod:<9} "
                f"{Colors.DIM if not reason else Colors.RED}{reason or '—':<40}{Colors.END}"
            )

    # Recent media entries across all UIDs
    media = query_multi_uid_media(db_path, uids, limit, lookback_hours)
    print(f"\n  {Colors.CYAN}── Recent Media Entries ({len(media)}) ──{Colors.END}")
    if not media:
        print(f"  {Colors.DIM}(no media entries found){Colors.END}")
    else:
        for m in media:
            fname = Path(m["file_path"]).name if m["file_path"] else "?"
            verified = yes_no(m["verified"])
            uploaded = yes_no(m["uploaded"])
            rewarded = yes_no(m["rewarded"])
            mod = m["modality"] or "?"
            c2pa = yes_no(m["c2pa_verified"])
            ts = fmt_ts(m["created_at"])
            ago = fmt_ago(m["created_at"])

            print(f"  ┌─ UID {m['uid']} {Colors.BOLD}{fname}{Colors.END}")
            print(f"  │  {Colors.DIM}{ts} ({ago}){Colors.END}")
            print(f"  │  Modality: {mod}  │  Verified: {verified}  │  Uploaded: {uploaded}  │  Rewarded: {rewarded}  │  C2PA: {c2pa}")
            if m.get("c2pa_issuer"):
                print(f"  │  C2PA Issuer: {m['c2pa_issuer']}")
            if m.get("file_size"):
                sz = m["file_size"]
                if sz < 1024:
                    sz_str = f"{sz}B"
                elif sz < 1024 * 1024:
                    sz_str = f"{sz / 1024:.1f}KB"
                else:
                    sz_str = f"{sz / (1024 * 1024):.1f}MB"
                print(f"  │  Size: {sz_str}  │  Format: {m.get('format', '?')}")
            print(f"  └─")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def list_failed_media(base_dir: Path, uid_filter: int | None = None) -> None:
    """Walk the failed_media/ directory and list stored rejection artifacts."""
    failed_dir = base_dir / "failed_media"
    if not failed_dir.exists():
        print(f"\n  {Colors.DIM}No failed_media/ directory found at {failed_dir}{Colors.END}")
        print(f"  {Colors.DIM}Enable --store-failed-media on the validator to collect rejected media.{Colors.END}")
        return

    # Structure: failed_media/<uid>/<reason_slug>/<task_id>.<ext>
    entries: list[dict] = []
    for uid_dir in sorted(failed_dir.iterdir()):
        if not uid_dir.is_dir():
            continue
        try:
            uid = int(uid_dir.name)
        except ValueError:
            continue
        if uid_filter is not None and uid != uid_filter:
            continue
        for reason_dir in sorted(uid_dir.iterdir()):
            if not reason_dir.is_dir():
                continue
            for f in sorted(reason_dir.iterdir()):
                if f.is_file():
                    sz = f.stat().st_size
                    entries.append({
                        "uid": uid,
                        "reason": reason_dir.name.replace("_", " "),
                        "filename": f.name,
                        "path": str(f),
                        "size": sz,
                    })

    if not entries:
        filter_msg = f" for UID {uid_filter}" if uid_filter is not None else ""
        print(f"\n  {Colors.DIM}No failed media found{filter_msg}.{Colors.END}")
        return

    print(f"\n{Colors.BOLD}{Colors.HEADER}🗑️  Failed Verification Media{Colors.END}")
    print(f"   {Colors.DIM}{failed_dir}{Colors.END}\n")

    # Group by UID + reason
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[f"UID {e['uid']}"].append(e)

    for group_key in sorted(groups.keys(), key=lambda x: int(x.split()[1])):
        group_entries = groups[group_key]
        print(f"  {Colors.BOLD}{Colors.CYAN}{group_key}{Colors.END}  ({len(group_entries)} files)")
        for e in group_entries:
            sz_str = (
                f"{e['size']}B" if e['size'] < 1024
                else f"{e['size']/1024:.1f}KB" if e['size'] < 1024*1024
                else f"{e['size']/(1024*1024):.1f}MB"
            )
            print(f"    {Colors.RED}{e['reason']:<45}{Colors.END}  {sz_str:>8}  {Colors.DIM}{e['filename']}{Colors.END}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Miner verification & challenge statistics"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="~/.cache/sn34/prompts.db",
        help="Path to the prompt database (default: ~/.cache/sn34/prompts.db)",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Show detailed stats for a specific miner UID",
    )
    parser.add_argument(
        "--coldkey",
        type=str,
        default=None,
        help="Show detailed stats for all miners under a coldkey (SS58 address)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent challenge/media entries to show in --uid mode (default: 20)",
    )
    parser.add_argument(
        "--lookback-hours",
        "--since",
        type=float,
        default=None,
        dest="lookback_hours",
        help="Only consider records from the last N hours",
    )
    parser.add_argument(
        "--by-coldkey",
        action="store_true",
        default=False,
        help="Group the summary table by coldkey (queries chain metagraph)",
    )
    parser.add_argument(
        "--chain-endpoint",
        type=str,
        default=None,
        help="Subtensor chain endpoint (default: finney network); only used with --by-coldkey or --uid",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        help="Subnet UID for metagraph lookup (default: auto-detect from SN34/SN379)",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        default=False,
        help="List stored failed-verification media (requires --store-failed-media on validator)",
    )

    args = parser.parse_args()
    db_path = Path(args.db_path).expanduser()

    # --failed mode: list stored rejection artifacts (no DB needed)
    if args.failed:
        base_dir = db_path.parent
        list_failed_media(base_dir, uid_filter=args.uid)
        print(f"\n{Colors.GREEN}✅ Failed media listing complete!{Colors.END}")
        return

    if not db_path.exists():
        print(f"{Colors.RED}❌ Database not found: {db_path}{Colors.END}")
        print(
            "💡 The database will be created automatically when the validator first runs."
        )
        sys.exit(1)

    print(f"{Colors.CYAN}📍 Database: {db_path}{Colors.END}")

    # Resolve netuid
    netuid = args.netuid or 34

    # All modes need the summary rows first
    rows = query_miner_summaries(db_path, args.lookback_hours)

    # Try fetching coldkeys if --by-coldkey or --coldkey was requested
    need_metagraph = args.by_coldkey or args.coldkey is not None
    coldkey_map: dict[int, str] | None = None
    if need_metagraph:
        coldkey_map = resolve_coldkeys(rows, netuid, args.chain_endpoint)
        if not coldkey_map:
            # Fall back to flat view
            print_miner_summary_table(rows, args.lookback_hours)
            totals = query_grand_totals(db_path)
            print_grand_totals(totals)
            print(f"\n{Colors.GREEN}✅ Miner stats complete!{Colors.END}")
            return

    if args.coldkey is not None and coldkey_map:
        # Find all UIDs belonging to this coldkey
        target = args.coldkey.strip()
        matching_uids = sorted(uid for uid, ck in coldkey_map.items() if ck == target)
        if not matching_uids:
            print(f"\n  {Colors.YELLOW}⚠️  No miners found for coldkey {fmt_coldkey(target)}{Colors.END}")
            return
        print_coldkey_detail(
            coldkey=target,
            uids=matching_uids,
            rows=rows,
            db_path=db_path,
            limit=args.limit,
            lookback_hours=args.lookback_hours,
        )
    elif args.uid is not None:
        # If we have a coldkey map, pass it; otherwise the detail view
        # just shows hotkey (no chain connection needed)
        print_miner_detail(
            uid=args.uid,
            db_path=db_path,
            limit=args.limit,
            lookback_hours=args.lookback_hours,
            coldkey_map=coldkey_map or {},
        )
    elif args.by_coldkey and coldkey_map:
        print_miner_summary_grouped(rows, coldkey_map, args.lookback_hours)
        totals = query_grand_totals(db_path)
        print_grand_totals(totals)
    else:
        print_miner_summary_table(rows, args.lookback_hours)
        totals = query_grand_totals(db_path)
        print_grand_totals(totals)

    print(f"\n{Colors.GREEN}✅ Miner stats complete!{Colors.END}")


if __name__ == "__main__":
    main()
