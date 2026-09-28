"""Bounded, read-only chain context for generator performance.

LastUpdate tracks submissions (commits on commit-reveal subnets), not reveals.
No wallet is opened and no extrinsics are submitted. RPC work runs in a child
process so a stalled endpoint cannot hang the CLI or contaminate JSON output.
"""

import json
import subprocess
import sys

PREFIX = "GAS_CHAIN_SNAPSHOT "
MAX_VALIDATORS = 5
# Avoid archive fallbacks for stale validators. Their exact block is still shown.
MAX_TIMESTAMP_AGE_BLOCKS = 7200


def _last_snapshot(output):
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    for line in reversed((output or "").splitlines()):
        if line.startswith(PREFIX):
            try:
                return json.loads(line[len(PREFIX):])
            except (ValueError, TypeError):
                continue
    return None


def fetch_generator_chain(hotkey, *, network="finney", netuid=34, timeout=30):
    """Return unknown/partial on failure; never turn an RPC failure into zero."""
    base = dict(status="unavailable", network=network, netuid=netuid,
                incentive=None, validators=[])
    try:
        result = subprocess.run(
            [sys.executable, "-m", __name__, hotkey, network, str(netuid)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        snapshot = _last_snapshot(result.stdout)
        if snapshot is not None:
            return snapshot
    except subprocess.TimeoutExpired as exc:
        snapshot = _last_snapshot(exc.stdout)
        if snapshot is not None:
            snapshot.update(status="partial", warning="Chain lookup timed out; partial snapshot shown.")
            return snapshot
        base["warning"] = "Chain lookup timed out; incentive is unknown, not zero."
        return base
    except OSError:
        pass
    base["warning"] = "Chain lookup unavailable; incentive is unknown, not zero."
    return base


def collect_snapshot(subtensor, hotkey, netuid, network, publish=lambda snapshot: None):
    """All values are read at one metagraph block; timestamps come from chain."""
    graph = subtensor.metagraph(netuid, lite=True)
    block = int(graph.block)
    snapshot = dict(status="partial", network=network, netuid=netuid, block=block,
                    incentive=None, validators=[], as_of=None)
    if hotkey not in graph.hotkeys:
        snapshot.update(status="not_registered", warning="Hotkey is not registered on this subnet.")
        return snapshot
    uid = graph.hotkeys.index(hotkey)
    snapshot.update(uid=uid, incentive=float(graph.I[uid]))
    publish(snapshot)
    try:
        snapshot["as_of"] = subtensor.get_timestamp(block=block).isoformat()
        snapshot["commit_reveal_enabled"] = bool(subtensor.commit_reveal_enabled(netuid, block=block))
        weights = dict(subtensor.weights(netuid, block=block))
        permitted = [i for i, permit in enumerate(graph.validator_permit) if bool(permit)]
        snapshot["validator_count"] = len(permitted)
        snapshot["positive_weight_count"] = sum(
            any(int(target) == uid and int(value) > 0 for target, value in weights.get(i, []))
            for i in permitted)
        chosen = sorted(permitted, key=lambda i: float(graph.S[i]), reverse=True)[:MAX_VALIDATORS]
        for i in chosen:
            entries = weights.get(i, [])
            total = sum(int(value) for _, value in entries)
            raw_weight = sum(int(value) for target, value in entries if int(target) == uid)
            last = int(graph.last_update[i])
            snapshot["validators"].append(dict(
                uid=i, hotkey=graph.hotkeys[i],
                weight=raw_weight / total if total else 0.0,
                submitted_block=last if 0 < last <= block else None,
                submitted_at=None,
                timing_note=(f">{MAX_TIMESTAMP_AGE_BLOCKS} blocks ago"
                             if last > 0 and block - last > MAX_TIMESTAMP_AGE_BLOCKS else None),
            ))
        publish(snapshot)
        timestamps = {}
        for validator in snapshot["validators"]:
            last = validator["submitted_block"]
            if last and block - last <= MAX_TIMESTAMP_AGE_BLOCKS:
                if last not in timestamps:
                    timestamps[last] = subtensor.get_timestamp(block=last).isoformat()
                validator["submitted_at"] = timestamps[last]
                publish(snapshot)
        snapshot["status"] = "ok"
    except Exception:
        snapshot["warning"] = "Some chain details are unavailable; missing timestamps are not estimated."
    return snapshot


def _main():
    import bittensor as bt
    hotkey, network, netuid = sys.argv[1:]

    def publish(snapshot):
        print(PREFIX + json.dumps(snapshot, allow_nan=False), flush=True)

    chain = None
    try:
        chain = bt.Subtensor(network=network, retry_forever=False)
        publish(collect_snapshot(chain, hotkey, int(netuid), network, publish))
    except Exception:
        publish(dict(status="unavailable", network=network, netuid=int(netuid),
                     incentive=None, validators=[], warning="Chain lookup unavailable; incentive is unknown, not zero."))
    finally:
        if chain is not None:
            chain.close()


if __name__ == "__main__":
    _main()
