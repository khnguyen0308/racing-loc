#!/usr/bin/env python3
"""STEP 1 - what is inside the mcap?  Prints every topic (type, count, rate) and ONE decoded
example of each non-image topic, so the team knows the exact field names.

    python3 tools_team/inspect_mcap.py data/c0_..._camloc_0.mcap  > notes/mcap_inspect.txt
"""
import json, sys
from pathlib import Path
from mcap.reader import make_reader

# Allow this file to be run directly from the repository root on all platforms.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools_team.mcap_utils import iter_msgs, to_dict  # noqa: E402


def main(path):
    with open(path, "rb") as f:
        s = make_reader(f).get_summary()
    st = s.statistics
    dur = (st.message_end_time - st.message_start_time) / 1e9
    print(f"duration {dur:.1f} s, {st.message_count} messages\n")
    print(f"{'topic':62s} {'type':42s} {'count':>8s} {'Hz':>7s}")
    topics = []
    for cid, ch in sorted(s.channels.items(), key=lambda kv: kv[1].topic):
        n = st.channel_message_counts.get(cid, 0)
        print(f"{ch.topic:62s} {s.schemas[ch.schema_id].name:42s} {n:8d} {n/dur:7.2f}")
        topics.append(ch.topic)
    print("\n=== one example per non-image topic ===")
    seen = set()
    want = [t for t in topics if "compressed_image" not in t]
    for topic, _, msg in iter_msgs(path, want):
        if topic in seen:
            continue
        seen.add(topic)
        print(f"\n--- {topic}")
        print(json.dumps(to_dict(msg), indent=1, default=str)[:6000])
        if len(seen) == len(want):
            break
    # one image header too
    for topic, _, msg in iter_msgs(path, [t for t in topics if "compressed_image" in t][:1]):
        print(f"\n--- {topic}\n", json.dumps(to_dict(msg), indent=1, default=str)[:1500])
        break


if __name__ == "__main__":
    main(sys.argv[1])
