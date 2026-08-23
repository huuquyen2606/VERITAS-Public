"""
Redis Data Inspector
====================
Dump and inspect all data stored for a specific file in Redis.

Usage:
    python inspect_redis.py --name sample.exe --type baseline
    python inspect_redis.py --name sample_adv.exe --type episode
    python inspect_redis.py --dump-all
"""

import redis
import json
import argparse
import sys

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0


def pretty_print_hash(r, key):
    """Pretty-print a Redis hash with formatted JSON values."""
    data = r.hgetall(key)
    if not data:
        print(f"  (Key '{key}' not found or empty)")
        return None
    
    print(f"\n{'='*70}")
    print(f"  KEY: {key}")
    print(f"  Fields: {len(data)}")
    print(f"{'='*70}")
    
    for field in sorted(data.keys()):
        value = data[field]
        
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                print(f"\n  [{field}] (list, {len(parsed)} items):")
                for i, item in enumerate(parsed[:10]):
                    print(f"    [{i}] {item}")
                if len(parsed) > 10:
                    print(f"    ... and {len(parsed) - 10} more items")
            elif isinstance(parsed, dict):
                print(f"\n  [{field}] (dict, {len(parsed)} keys):")
                print(f"    {json.dumps(parsed, indent=4)}")
            else:
                print(f"\n  [{field}]: {parsed}")
        except (json.JSONDecodeError, TypeError):
            display = value if len(value) <= 200 else value[:200] + f"... ({len(value)} chars)"
            print(f"\n  [{field}]: {display}")
    
    print(f"\n{'='*70}")
    return data


def dump_all_keys(r):
    """Find and dump all baseline and episode keys."""
    print("\n[*] Scanning for all pipeline keys in Redis...\n")
    
    baseline_keys = sorted(r.keys("baseline:*:data"))
    episode_keys = sorted(r.keys("episode:*:data"))
    barrier_keys = sorted(r.keys("*:tasks_remaining"))
    
    print(f"  Found {len(baseline_keys)} baseline keys")
    print(f"  Found {len(episode_keys)} episode keys")
    print(f"  Found {len(barrier_keys)} barrier keys")
    
    if barrier_keys:
        print(f"\n--- BARRIER STATUS ---")
        for bk in barrier_keys:
            val = r.get(bk)
            print(f"  {bk} = {val}")
    
    rr = r.get("cape_rr_index")
    if rr:
        print(f"\n  cape_rr_index = {rr}")
    
    for key in baseline_keys:
        pretty_print_hash(r, key)
    for key in episode_keys:
        pretty_print_hash(r, key)


def main():
    parser = argparse.ArgumentParser(description="Redis Data Inspector")
    parser.add_argument("--name", help="File name to inspect (e.g., sample.exe)")
    parser.add_argument("--type", choices=["baseline", "episode"], help="Key type")
    parser.add_argument("--dump-all", action="store_true", help="Dump ALL pipeline keys")
    parser.add_argument("--flush", action="store_true", help="Flush ALL pipeline keys (DANGEROUS)")
    args = parser.parse_args()

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    try:
        r.ping()
    except redis.ConnectionError:
        print("[!] Cannot connect to Redis.")
        sys.exit(1)

    if args.flush:
        confirm = input("[!] This will DELETE all baseline:* and episode:* keys. Type 'YES' to confirm: ")
        if confirm == "YES":
            for key in r.keys("baseline:*"):
                r.delete(key)
            for key in r.keys("episode:*"):
                r.delete(key)
            r.delete("cape_rr_index")
            print("[✓] All pipeline keys flushed.")
        else:
            print("[*] Cancelled.")
        return

    if args.dump_all:
        dump_all_keys(r)
        return

    if args.name and args.type:
        key = f"{args.type}:{args.name}:data"
        pretty_print_hash(r, key)
        
        barrier_key = f"{args.type}:{args.name}:tasks_remaining"
        val = r.get(barrier_key)
        if val:
            print(f"  Barrier ({barrier_key}): {val}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
