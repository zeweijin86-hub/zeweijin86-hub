### Zewei Jin · 金泽玮

Self-supervised video understanding — frozen backbones, attentive probes, and the routing between them.

---

## vjepa2-router

**Per-video classifier routing for Meta's V-JEPA 2**

[![tests](https://github.com/zeweijin86-hub/vjepa2-router/actions/workflows/tests.yml/badge.svg)](https://github.com/zeweijin86-hub/vjepa2-router/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/zeweijin86-hub/vjepa2-router/blob/main/LICENSE)
[![upstream](https://img.shields.io/badge/upstream-vjepa2%40204698b4-lightgrey)](https://github.com/facebookresearch/vjepa2)

Frozen-backbone probing on V-JEPA 2 trains `H` independent attentive probes, then keeps only the single best-scoring head at validation time — throwing away whatever the other heads learned. This overlay replaces that selection with a router: a small network that reads every probe's class probabilities for one clip and decides how much to trust each head, trained on exact mixture negative log-likelihood.

|  |  |
|---|---|
| **Loss** | FP32-safe focal loss, optional dynamic difficulty weighting |
| **Regularization** | non-target inter-probe diversity against detached peers, a no-op on the binary simplex |
| **Router** | LayerNorm MLP over `[H, C]` probabilities, trained with mixture NLL |
| **Engineering** | DDP gather + dedupe, stratified same-split comparison, joint-atomic checkpoints, `--val_only` / `--infer_only` |
| **Verification** | 13 unit tests on Python 3.10–3.12 with CPU-only PyTorch |

**[Repository](https://github.com/zeweijin86-hub/vjepa2-router)** · [v0.1.0](https://github.com/zeweijin86-hub/vjepa2-router/releases/tag/v0.1.0) · [patch pinned to `204698b4`](https://github.com/zeweijin86-hub/vjepa2-router/blob/main/patches/eval_modifications.patch)

---

<sub>**Side project** — [Repo Time Machine](https://github.com/zeweijin86-hub/Repo-Time-Machine): a local-first Git history visualizer (React + TypeScript, static, nothing uploaded).</sub>
