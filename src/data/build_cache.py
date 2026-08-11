"""
Pre-build every Level-1 graph to data/graphs/ so training does not pay the ~15-35s
parse cost per config on every epoch.

    python -m src.data.build_cache                    # all designs, all usable configs
    python -m src.data.build_cache --designs Vortex-small
"""

import argparse
import time
from pathlib import Path

from . import circuitnet as C


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", type=Path, default=Path("data/external/circuitnet/timing_extracted")
    )
    p.add_argument("--designs", nargs="+", default=["Vortex-small", "nvdla-small"])
    p.add_argument("--out", type=Path, default=Path("data/graphs"))
    args = p.parse_args()

    for design in args.designs:
        ddir = args.root / design
        cache = args.out / design
        cache.mkdir(parents=True, exist_ok=True)
        configs = C.list_configs(ddir)
        print(f"=== {design}: {len(configs)} usable configs -> {cache}", flush=True)
        for i, cfg in enumerate(configs, 1):
            if (cache / f"{cfg}.pt").exists():
                continue
            t0 = time.time()
            g = C.load_circuitnet(ddir, cfg, cache_dir=cache)
            print(
                f"[{i:3d}/{len(configs)}] {cfg}  nodes={g.num_nodes:,} "
                f"edges={g.edge_index.shape[1]:,}  {time.time() - t0:.1f}s",
                flush=True,
            )
    print("cache build complete")


if __name__ == "__main__":
    main()
