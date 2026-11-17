"""Train PPO partial-rebalancing policies on the TRAINING span only.

Single process:   python scripts/08_train_ppo.py --cases etf:ridge --seeds 0
Multi-process:    scripts/run_ppo.sh main
Multi-node:       see scripts/launch_multinode_example.sh (same script, no code change)
Outputs (rank 0): tmp/rebalance/policies/{tag}/{dataset}_{signal}_s{seed}.pt and .json (training history, wall clock).
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, torch.distributed as dist
from qmb.rebalance import cases as data, env, ppo
from qmb.rebalance.splits import split_indices

ap = argparse.ArgumentParser()
ap.add_argument("--cases", nargs="*", default=None, help="dataset:signal; default all cached cases")
ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--updates", type=int, default=ppo.HP["updates"])
ap.add_argument("--tag", default="main")
ap.add_argument("--resume", action="store_true", help="skip finished checkpoints for this configuration")
a = ap.parse_args()

if "RANK" in os.environ:
    dist.init_process_group("gloo")
rank = dist.get_rank() if dist.is_initialized() else 0
world = dist.get_world_size() if dist.is_initialized() else 1
torch.set_num_threads(1)
t0 = time.perf_counter()
log = lambda *x: print(f"[{time.perf_counter()-t0:6.1f}s r{rank}/{world}]", *x, flush=True) if rank == 0 else None

cases = a.cases or sorted(p.stem[5:].replace("_", ":", 1) for p in data.TMP.glob("case_*.npz"))
out = data.TMP / "policies" / a.tag; out.mkdir(parents=True, exist_ok=True)
hp = dict(ppo.HP, updates=a.updates)
for cs in cases:
    ds, sig = cs.split(":")
    case = data.load_case(data.TMP / f"case_{ds}_{sig}.npz")
    lo, hi, te0, te1 = split_indices(case)
    mkt = env.Market(case, np.arange(lo, hi))
    fac = lambda B, rng: env.BatchEnv(mkt, lo, hi, hp["L"], B, rng, hp["risk_aversion"])
    for seed in a.seeds:
        stem = out / f"{ds}_{sig}_s{seed}"
        model_path, meta_path = stem.with_suffix(".pt"), stem.with_suffix(".json")
        if a.resume and model_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            if meta["world_size"] != world or meta["hp"] != hp:
                raise ValueError(f"Existing checkpoint configuration does not match {stem}")
            log(f"{ds} {sig} seed {seed}: skipped complete run")
            continue
        ts = time.perf_counter()
        log(f"{ds} {sig} seed {seed}: train weeks [{lo},{hi}) world={world}")
        model, hist = ppo.train(fac, len(env.FEATURES), seed, hp, log=log)
        wall = time.perf_counter() - ts
        if rank == 0:
            tmp_model = model_path.with_suffix(".pt.part")
            torch.save(model.state_dict(), tmp_model)
            tmp_model.replace(model_path)
            meta = {"dataset": ds, "signal": sig, "seed": seed, "world_size": world, "wall_seconds": wall,
                    "hp": hp, "train_window": [int(lo), int(hi)], "history": hist}
            tmp_meta = meta_path.with_suffix(".json.part")
            tmp_meta.write_text(json.dumps(meta, indent=0) + "\n")
            tmp_meta.replace(meta_path)
            log(f"{ds} {sig} seed {seed}: done in {wall:.1f}s")
if dist.is_initialized():
    dist.barrier(); dist.destroy_process_group()
