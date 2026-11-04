#!/usr/bin/env python
"""训练行为克隆基线。

    python scripts/train_bc.py --data data/demos --out runs/bc --epochs 30
    python scripts/train_bc.py --data data/demos --image-size 128 --batch-size 32

输出: ``runs/bc/policy.pt``（含归一化统计与词表）+ ``runs/bc/metrics.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vla_vehicle.config import TrainConfig                     # noqa: E402
from vla_vehicle.data.dataset import EpisodeDataset, make_loader  # noqa: E402
from vla_vehicle.policies.bc import BCNet, BCPolicy            # noqa: E402
from vla_vehicle.sim.objects import Tokenizer                  # noqa: E402


def pick_device(name: str) -> str:
    import torch
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--all-failed", action="store_true", help="连失败回合一起训练")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    for k in ("data", "out", "epochs", "batch_size", "lr", "image_size", "stride", "device"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.all_failed:
        cfg.only_success = False

    import torch
    import torch.nn.functional as F

    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ds = EpisodeDataset(cfg.data, only_success=cfg.only_success, stride=cfg.stride,
                        image_size=cfg.image_size)
    stats = ds.stats()
    tok = Tokenizer()
    tr, va = ds.split(cfg.val_ratio, seed=cfg.seed)
    print(f"数据集: {ds.summary()}")
    print(f"训练帧 {len(tr)} / 验证帧 {len(va)} | 词表 {tok.size} | 设备 {device}")

    net = BCNet.build(vocab_size=tok.size, image_size=cfg.image_size)
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr_loader = make_loader(tr, cfg.batch_size, True, device, cfg.num_workers)
    va_loader = make_loader(va, min(cfg.batch_size, 64), False, device, 0) if len(va) else None

    def to_device(b):
        return {k: v.to(device, non_blocking=True) for k, v in b.items()}

    def norm_batch(b):
        b = dict(b)
        b["q"] = (b["q"] - torch.as_tensor(stats["q_mean"], device=device)) \
            / torch.as_tensor(stats["q_std"], device=device)
        b["dq"] = b["dq"] / torch.as_tensor(stats["dq_std"], device=device)
        b["tcp"] = (b["tcp"] - torch.as_tensor(stats["tcp_mean"], device=device)) \
            / torch.as_tensor(stats["tcp_std"], device=device)
        return b

    hist = []
    best = {"val": float("inf"), "epoch": -1}
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        tot, n = 0.0, 0
        for batch in tr_loader:
            batch = norm_batch(to_device(batch))
            pos, logit = net(batch)
            loss_pos = F.l1_loss(pos, batch["action"][:, :3])
            loss_grip = F.binary_cross_entropy_with_logits(logit, batch["action"][:, 3])
            loss = loss_pos + 0.5 * loss_grip
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(batch["grip"])
            n += len(batch["grip"])
        tr_loss = tot / max(1, n)

        va_loss = float("nan")
        acc = float("nan")
        if va_loader is not None:
            net.eval()
            tot, n, correct = 0.0, 0, 0
            with torch.no_grad():
                for batch in va_loader:
                    batch = norm_batch(to_device(batch))
                    pos, logit = net(batch)
                    loss_pos = F.l1_loss(pos, batch["action"][:, :3])
                    loss_grip = F.binary_cross_entropy_with_logits(logit, batch["action"][:, 3])
                    tot += float(loss_pos + 0.5 * loss_grip) * len(batch["grip"])
                    n += len(batch["grip"])
                    pred = (torch.sigmoid(logit) >= 0.5).float()
                    correct += int((pred == (batch["action"][:, 3] >= 0.5).float()).sum())
            va_loss = tot / max(1, n)
            acc = correct / max(1, n)
            if va_loss < best["val"]:
                best = {"val": va_loss, "epoch": epoch}
        hist.append(dict(epoch=epoch, train=tr_loss, val=va_loss, grip_acc=acc))
        print(f"epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"grip_acc {acc:.3f}", flush=True)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = BCPolicy(net, stats, tok, device=device, image_size=cfg.image_size)
    policy.save(out / "policy.pt", extra=dict(best=best, hist=hist, data=str(cfg.data)))
    (out / "metrics.json").write_text(json.dumps(
        dict(best=best, hist=hist, dataset=ds.summary(), config=cfg.__dict__),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最佳验证损失 {best['val']:.4f} (epoch {best['epoch']}), "
          f"用时 {time.time() - t0:.1f}s")
    print(f"模型: {(out / 'policy.pt').resolve()}")
    print("下一步: python scripts/eval_policy.py --ckpt " + str(out / "policy.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
