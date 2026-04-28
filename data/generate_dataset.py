"""
generate_dataset.py — generate a BALANCED dataset.json

Run from project root:
    python data/generate_dataset.py

Produces 200 timesteps x 15 nodes with balanced labels:
  - SAFE    (0)  water_level < 25 cm
  - WARNING (1)  25 <= water_level < 35 cm
  - FLOOD   (2)  water_level >= 35 cm

5 rain phases, 40 steps each:
  Phase 0 (t   0-39 ): dry       -> SAFE
  Phase 1 (t  40-79 ): rising    -> WARNING / early FLOOD
  Phase 2 (t  80-119): heavy     -> FLOOD
  Phase 3 (t 120-159): draining  -> WARNING / late FLOOD
  Phase 4 (t 160-199): recovery  -> SAFE

Nodes each have a random personal water-level offset (+/-8 cm)
so within every phase the 15 nodes naturally span 2-3 risk classes.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

random.seed(37)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_NODES       = 15
NUM_TIMESTEPS   = 200
STEPS_PER_PHASE = 40
TIME_INTERVAL   = 10           # minutes between readings
START_TIME      = datetime(2026, 4, 27)

SAFE_MAX    = 25.0
WARNING_MAX = 35.0

NODES = [f"N{i}" for i in range(1, NUM_NODES + 1)]

EDGES = [
    ("N1",  "N3"),  ("N2",  "N3"),
    ("N3",  "N7"),
    ("N4",  "N6"),  ("N5",  "N6"),
    ("N6",  "N8"),
    ("N10", "N11"), ("N11", "N13"), ("N12", "N13"),
    ("N7",  "N9"),  ("N8",  "N9"),
    ("N13", "N14"), ("N9",  "N14"),
    ("N14", "N15"),
]

OUTPUT_PATH = Path(__file__).parent / "dataset.json"

# Phase definitions: (wl_center, rain_base)
# Node offset of +-8 cm means the 15 nodes will naturally straddle thresholds.
PHASES = [
    dict(center=14.0, rain=1.5),   # Phase 0: dry      -> SAFE
    dict(center=30.0, rain=6.5),   # Phase 1: rising   -> WARNING (some FLOOD)
    dict(center=45.0, rain=12.0),  # Phase 2: flood    -> FLOOD (some WARNING)
    dict(center=30.0, rain=3.0),   # Phase 3: draining -> WARNING (some FLOOD)
    dict(center=14.0, rain=1.5),   # Phase 4: recovery -> SAFE
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def flood_risk_label(wl: float) -> int:
    if wl < SAFE_MAX:
        return 0
    elif wl < WARNING_MAX:
        return 1
    return 2


def build_adjacency_matrix():
    idx = {n: i for i, n in enumerate(NODES)}
    mat = [[0] * NUM_NODES for _ in range(NUM_NODES)]
    for src, dst in EDGES:
        mat[idx[src]][idx[dst]] = 1
    return mat


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate():
    # Each node has a fixed personal offset so the 15 nodes are always spread
    # across different risk classes within the same phase.
    node_offsets = {n: random.uniform(-8.0, 8.0) for n in NODES}

    # Lerp speed: how quickly each node's water level tracks its phase target
    lerp_speed = {n: random.uniform(0.20, 0.45) for n in NODES}

    # Start all nodes in the SAFE zone
    water = {n: random.uniform(10.0, 20.0) for n in NODES}

    records = []

    for t in range(NUM_TIMESTEPS):
        phase_idx = min(t // STEPS_PER_PHASE, len(PHASES) - 1)
        ph        = PHASES[phase_idx]
        timestamp = START_TIME + timedelta(minutes=t * TIME_INTERVAL)

        node_records = []
        for n in NODES:
            target  = ph["center"] + node_offsets[n]
            target  = max(4.0, target)

            prev_wl = water[n]
            new_wl  = prev_wl + lerp_speed[n] * (target - prev_wl)
            new_wl += random.gauss(0, 0.01)          # small noise
            new_wl  = max(4.0, min(70.0, new_wl))   # absolute clamp

            rate_of_rise = round(new_wl - prev_wl, 3)
            water[n]     = new_wl

            # Sensor readings derived from water level and rain phase
            rain_base     = ph["rain"] + random.gauss(0, 1.2)
            rainfall      = max(0.0, rain_base)
            humidity      = 40.0 + phase_idx * 9.0 + random.uniform(-3, 3)
            humidity      = max(40.0, min(95.0, humidity))
            cloud_density = 50.0 + rainfall * 1.5 + random.uniform(-4, 4)
            cloud_density = max(50.0, min(100.0, cloud_density))

            node_records.append({
                "node_id":       n,
                "rainfall":      round(rainfall, 6),
                "humidity":      round(humidity, 6),
                "cloud_density": round(cloud_density, 6),
                "water_level":   round(new_wl, 2),
                "rate_of_rise":  rate_of_rise,
                "flood_risk":    flood_risk_label(new_wl),
            })

        records.append({
            "time_step": t,
            "timestamp": timestamp.isoformat(),
            "nodes":     node_records,
        })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating balanced dataset ...")
    records = simulate()

    # Overall distribution
    counts = Counter()
    for rec in records:
        for nd in rec["nodes"]:
            counts[nd["flood_risk"]] += 1
    total  = sum(counts.values())
    labels = {0: "SAFE", 1: "WARNING", 2: "FLOOD"}

    print(f"\n{'Label':<10} {'Count':>6}  {'%':>6}")
    print("-" * 28)
    for k in sorted(counts):
        print(f"{labels[k]:<10} {counts[k]:>6}  {counts[k]/total*100:>5.1f}%")

    # Per-phase breakdown
    print(f"\nPer-phase label counts ({STEPS_PER_PHASE} steps x {NUM_NODES} nodes each):")
    for p in range(len(PHASES)):
        pc = Counter()
        s, e = p * STEPS_PER_PHASE, (p + 1) * STEPS_PER_PHASE
        for rec in records[s:e]:
            for nd in rec["nodes"]:
                pc[nd["flood_risk"]] += 1
        print(f"  Phase {p}: SAFE={pc[0]:>4} WARN={pc[1]:>4} FLOOD={pc[2]:>4}")

    dataset = {
        "metadata": {
            "num_nodes":             NUM_NODES,
            "num_timesteps":         NUM_TIMESTEPS,
            "time_interval_minutes": TIME_INTERVAL,
            "total_duration_hours":  round(NUM_TIMESTEPS * TIME_INTERVAL / 60, 2),
            "start_time":            START_TIME.isoformat(),
            "created_at":            datetime.utcnow().isoformat(),
            "description": (
                "Balanced flood prediction dataset. "
                "5 phases (40 steps each): dry->rising->flood->draining->recovery. "
                "Node offsets ensure mixed labels within each phase."
            ),
            "thresholds": {
                "safe_max_water_level":    SAFE_MAX,
                "warning_max_water_level": WARNING_MAX,
            },
        },
        "graph": {
            "nodes":            NODES,
            "edges":            [f"{s} -> {d}" for s, d in EDGES],
            "num_edges":        len(EDGES),
            "adjacency_matrix": build_adjacency_matrix(),
        },
        "data": records,
    }

    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"\nDataset written to : {OUTPUT_PATH}")
    print(f"Timesteps : {NUM_TIMESTEPS}   Nodes : {NUM_NODES}")


if __name__ == "__main__":
    main()
