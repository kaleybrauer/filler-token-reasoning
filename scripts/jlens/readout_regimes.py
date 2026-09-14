"""Do J-lens readouts change between the batch-1 and fit-regime activations?

Same readout as workspace_signatures.py panel (a): z = (h @ J.T) @ U.T with
U = lm_head * rms_norm_weight, scored against the model's own final-layer top-1
(computed from the same regime's layer-60 state). Per source layer, for the prompts
both states files share:
  agree_jlens   fraction of (prompt, position) whose J-lens top-1 is identical
  agree_logit   same for the plain logit lens (h @ U.T), the regime floor without J
  model_agree   fraction where the model's own final-layer top-1 is identical
  acc_b1/acc_fit panel (a) top-1 accuracy in each regime

    python readout_regimes.py <batch1_states.pt> <fitregime_states.pt> <lens.pt> [device] [n_prompts] [layers]
"""
import sys
import time

import numpy as np
import torch

b1_path, fr_path, lens_path = sys.argv[1:4]
dev = sys.argv[4] if len(sys.argv) > 4 else "cuda:0"
n_max = int(sys.argv[5]) if len(sys.argv) > 5 else None
W = "/workspace/filler-token-reasoning/data/model_weights/deepseek_v3"

t0 = time.time()
fr = torch.load(fr_path, map_location="cpu", weights_only=False, mmap=True)
b1 = torch.load(b1_path, map_location="cpu", weights_only=False, mmap=True)
lo_fr, lo_b1 = fr["prompt_span"][0], b1["prompt_span"][0]
n = min(fr["H"].shape[0], b1["H"].shape[0] - (lo_fr - lo_b1))
if n_max:
    n = min(n, n_max)
off = lo_fr - lo_b1
assert off >= 0 and torch.equal(fr["input_ids"][:n], b1["input_ids"][off:off + n]), "prompts differ"
lens = torch.load(lens_path, map_location="cpu", weights_only=False, mmap=True)
layers = ([int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6
          else sorted(lens["J"].keys()))
U = torch.from_numpy(np.load(f"{W}/lm_head_weight.npy").astype(np.float32)
                     * np.load(f"{W}/rms_norm_weight.npy").astype(np.float32)[None, :]).to(dev)
Hb = b1["H"][off:off + n]
Hf = fr["H"][:n]
final = fr["n_layers"] - 1


def top1(h, J=None, chunk=1024):
    out = []
    for s in range(0, h.shape[0], chunk):
        x = h[s:s + chunk].to(dev).float()
        if J is not None:
            x = x @ J.T
        out.append((x @ U.T).argmax(1).cpu())
    return torch.cat(out)


flat = lambda H, L: H[:, :, L].reshape(-1, H.shape[-1])
mb, mf = top1(flat(Hb, final)), top1(flat(Hf, final))
print(f"{n} prompts x {Hb.shape[1]} positions; regime {fr.get('regime')}; lens {lens_path.split('/')[-1]}")
print(f"model's own final-layer top-1 identical across regimes: {(mb == mf).float().mean():.4f}")
print(f"{'L':>3} {'agree_jlens':>11} {'agree_logit':>11} {'acc_b1':>7} {'acc_fit':>7}")
rows = []
for L in layers:
    J = lens["J"][L].to(dev).float()
    jb, jf = top1(flat(Hb, L), J), top1(flat(Hf, L), J)
    gb, gf = top1(flat(Hb, L)), top1(flat(Hf, L))
    r = dict(layer=L, agree_jlens=float((jb == jf).float().mean()),
             agree_logit=float((gb == gf).float().mean()),
             acc_b1=float((jb == mb).float().mean()), acc_fit=float((jf == mf).float().mean()))
    rows.append(r)
    print(f"{L:>3} {r['agree_jlens']:11.4f} {r['agree_logit']:11.4f} {r['acc_b1']:7.4f} {r['acc_fit']:7.4f}", flush=True)
    del J
a = np.array([[r["agree_jlens"], r["agree_logit"], r["acc_b1"], r["acc_fit"]] for r in rows])
print(f"mean over layers: agree_jlens {a[:, 0].mean():.4f}  agree_logit {a[:, 1].mean():.4f}  "
      f"acc_b1 {a[:, 2].mean():.4f}  acc_fit {a[:, 3].mean():.4f}   ({time.time() - t0:.0f}s)")
