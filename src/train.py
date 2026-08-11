"""
Training loop for Level-1 (and later Level-2 / the single-stage baseline).

    python -m src.train --protocol unseen-config --epochs 30
    python -m src.train --protocol unseen-design --epochs 30

Two evaluation protocols, per paper section 5.1:
  unseen-config  train and test on the same design, disjoint PPA configurations
  unseen-design  train on one design, test on a design never seen in training

Every reported number is accompanied by the IDENTITY BASELINE on the same nets -- the
pre-route estimate used unchanged. On this data that baseline already scores R^2 ~ 0.84,
so a model is only interesting insofar as it beats it. Reporting model R^2 alone would
be close to meaningless.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .data import feature_spec as fs
from .data.partition import partition_report, spatial_partition
from .models.stage1_model import Stage1Model


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def load_cached(
    cache_root: Path, designs: List[str], raw_root: Path, allow_partial: bool = False
) -> Dict[str, List[Path]]:
    """
    Load the cached graph list, and refuse to train on a HALF-BUILT cache.

    build_cache can be interrupted, and a partial cache trains fine while quietly
    reporting results for a fraction of the data. Compare against what the raw dataset
    actually offers and stop unless --allow-partial-cache is given.
    """
    from .data.circuitnet import list_configs

    out, partial = {}, []
    for d in designs:
        files = sorted((cache_root / d).glob("*.pt"))
        if not files:
            continue
        out[d] = files
        raw_dir = raw_root / d
        if raw_dir.exists():
            expected = len(list_configs(raw_dir))
            if len(files) < expected:
                partial.append(f"  {d}: {len(files)} cached of {expected} usable configs")

    if not out:
        raise SystemExit(
            f"no cached graphs under {cache_root}. Run: python -m src.data.build_cache"
        )
    if partial:
        msg = "cache is incomplete:\n" + "\n".join(partial)
        if not allow_partial:
            raise SystemExit(
                msg
                + "\n\nFinish it with:  python -m src.data.build_cache"
                + "\n(it skips configs already cached, so it resumes where it stopped)"
                + "\nOr pass --allow-partial-cache to train on what exists anyway."
            )
        print(f"WARNING: {msg}\n  proceeding because --allow-partial-cache was given")
    return out


def split(
    cached: Dict[str, List[Path]], protocol: str, seed: int = 0
) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = random.Random(seed)
    if protocol == "unseen-config":
        files = sorted(sum(cached.values(), []))
        rng.shuffle(files)
        n = len(files)
        n_tr, n_va = int(0.7 * n), int(0.15 * n)
        return files[:n_tr], files[n_tr : n_tr + n_va], files[n_tr + n_va :]

    if protocol == "unseen-design":
        designs = sorted(cached)
        if len(designs) < 2:
            raise SystemExit("unseen-design needs at least two cached designs")
        test_design = designs[-1]
        train = sorted(sum((v for k, v in cached.items() if k != test_design), []))
        rng.shuffle(train)
        n_va = max(1, int(0.15 * len(train)))
        return train[n_va:], train[:n_va], sorted(cached[test_design])

    raise SystemExit(f"unknown protocol {protocol}")


class Normalizer:
    """z-score, fitted on TRAIN graphs only and then frozen."""

    def __init__(self) -> None:
        self.x_mean = self.x_std = self.e_mean = self.e_std = None

    def fit(self, files: List[Path], max_graphs: int = 12) -> "Normalizer":
        xs, es = [], []
        for f in files[:max_graphs]:
            g = torch.load(f, weights_only=False)
            xs.append(g.x)
            es.append(g.edge_attr[:: max(1, g.edge_attr.shape[0] // 200_000)])
        x, e = torch.cat(xs), torch.cat(es)
        self.x_mean, self.x_std = x.mean(0), x.std(0).clamp_min(1e-6)
        self.e_mean, self.e_std = e.mean(0), e.std(0).clamp_min(1e-6)
        return self

    def __call__(self, g):
        g.x = (g.x - self.x_mean) / self.x_std
        g.edge_attr = (g.edge_attr - self.e_mean) / self.e_std
        return g

    def state_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("x_mean", "x_std", "e_mean", "e_std")}


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def r2(pred: Tensor, true: Tensor) -> Tensor:
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot.clamp_min(1e-12)


@torch.no_grad()
def evaluate(model, files, norm, args, device) -> dict:
    """
    Pooled metrics over every net in `files`, plus the identity baseline on the same nets.

    The baseline is the pre-route estimate itself, i.e. node feature columns
    pre_net_delay_0..3 -- read back through the normalizer so the comparison is against
    the true raw values, not the z-scored ones.
    """
    model.eval()
    names = fs.active_names(fs.LEVEL1_NODE, "A")
    pre_cols = [names.index(f"pre_net_delay_{c}") for c in range(fs.NET_DELAY_CHANNELS)]
    x_col, y_col = names.index("driver_x"), names.index("driver_y")

    preds, trues, bases = [], [], []
    for f in files:
        g = norm(torch.load(f, weights_only=False))
        for part in spatial_partition(g, args.parts, x_col, y_col):
            part = part.to(device)
            out = model(part.x, part.edge_index, part.edge_attr)
            preds.append(out.cpu())
            trues.append(part.y.cpu())
            # undo normalisation on the pre-route columns to recover the raw estimate
            base = part.x[:, pre_cols].cpu() * norm.x_std[pre_cols] + norm.x_mean[pre_cols]
            bases.append(base)
        del g

    pred, true, base = torch.cat(preds), torch.cat(trues), torch.cat(bases)
    return {
        "n_nets": int(true.shape[0]),
        "model_r2": r2(pred, true).tolist(),
        "model_mae": (pred - true).abs().mean(0).tolist(),
        "model_mse": ((pred - true) ** 2).mean(0).tolist(),
        "baseline_r2": r2(base, true).tolist(),
        "baseline_mae": (base - true).abs().mean(0).tolist(),
        "baseline_mse": ((base - true) ** 2).mean(0).tolist(),
    }


def fmt(m: dict) -> str:
    mr, br = np.mean(m["model_r2"]), np.mean(m["baseline_r2"])
    mm, bm = np.mean(m["model_mae"]), np.mean(m["baseline_mae"])
    return (
        f"R2 {mr:.4f} (base {br:.4f}, {mr - br:+.4f})   "
        f"MAE {mm:.4f} (base {bm:.4f}, {mm - bm:+.4f})"
    )


# --------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=Path("data/graphs"))
    p.add_argument(
        "--raw", type=Path, default=Path("data/external/circuitnet/timing_extracted")
    )
    p.add_argument("--designs", nargs="+", default=["Vortex-small", "nvdla-small"])
    p.add_argument(
        "--allow-partial-cache",
        action="store_true",
        help="train even if build_cache has not finished (results cover less data)",
    )
    p.add_argument(
        "--protocol", default="unseen-config", choices=["unseen-config", "unseen-design"]
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--parts", type=int, default=16, help="spatial tiles per graph")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=0, help="cap train graphs (0 = all)")
    p.add_argument("--max-eval", type=int, default=8, help="cap val/test graphs")
    p.add_argument("--out", type=Path, default=Path("results/trained_models"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cached = load_cached(args.cache, args.designs, args.raw, args.allow_partial_cache)
    train_f, val_f, test_f = split(cached, args.protocol, args.seed)
    if args.max_train:
        train_f = train_f[: args.max_train]
    val_f, test_f = val_f[: args.max_eval], test_f[: args.max_eval]
    print(
        f"protocol={args.protocol}  train={len(train_f)} val={len(val_f)} "
        f"test={len(test_f)} graphs  device={device}"
    )

    norm = Normalizer().fit(train_f)
    names = fs.active_names(fs.LEVEL1_NODE, "A")
    x_col, y_col = names.index("driver_x"), names.index("driver_y")

    probe = norm(torch.load(train_f[0], weights_only=False))
    rep = partition_report(probe, spatial_partition(probe, args.parts, x_col, y_col))
    print(
        f"partitioning: {rep['n_parts']} tiles, "
        f"node retention {rep['node_retention']:.3f}, "
        f"edge retention {rep['edge_retention']:.3f}, "
        f"largest tile {rep['max_part_nodes']:,} nodes / {rep['max_part_edges']:,} edges"
    )
    del probe

    model = Stage1Model(
        in_dim=fs.active_dim(fs.LEVEL1_NODE, "A"),
        edge_dim=fs.active_dim(fs.LEVEL1_EDGE, "A"),
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"{model.__class__.__name__}: {n_params:,} parameters, in_dim={model.in_dim}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        order = list(train_f)
        random.shuffle(order)
        for f in order:
            g = norm(torch.load(f, weights_only=False))
            parts = spatial_partition(g, args.parts, x_col, y_col)
            random.shuffle(parts)
            for part in parts:
                part = part.to(device)
                opt.zero_grad()
                out = model(part.x, part.edge_index, part.edge_attr)
                loss = F.mse_loss(out, part.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                nb += 1
            del g, parts

        val = evaluate(model, val_f, norm, args, device)
        val_mse = float(np.mean(val["model_mse"]))
        history.append({"epoch": epoch, "train_loss": tot / max(nb, 1), **val})
        print(
            f"epoch {epoch:3d}  train_mse {tot / max(nb, 1):.4f}  "
            f"val {fmt(val)}  {time.time() - t0:.0f}s",
            flush=True,
        )

        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "norm": norm.state_dict(),
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "feature_names": names,
                    "epoch": epoch,
                },
                args.out / f"stage1_{args.protocol}.pt",
            )

    test = evaluate(model, test_f, norm, args, device)
    print(f"\nTEST ({test['n_nets']:,} nets)  {fmt(test)}")
    print("  per-channel model R2   ", [round(v, 4) for v in test["model_r2"]])
    print("  per-channel baseline R2", [round(v, 4) for v in test["baseline_r2"]])

    (args.out / f"stage1_{args.protocol}_history.json").write_text(
        json.dumps({"history": history, "test": test}, indent=2)
    )


if __name__ == "__main__":
    main()
