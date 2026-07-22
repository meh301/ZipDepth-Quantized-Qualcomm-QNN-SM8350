"""Re-export ZipDepth with the TRAINED GlobalContextBlock attention pooling.

Upstream scripts/export.py:69-73 replaces the trained forward

    mask    = softmax(context_weight(x))      # learned attention over H*W
    context = bmm(x_flat, mask^T)
    return x + transform(context)

with `avg_pool2d`, which also discards the trained `context_weight` Conv2d.
Measured end-to-end that costs 24.1% of dynamic range raw / 2.74% RMSE after
affine alignment - more than int16 quantisation costs.

The substitution is avoidable: the bmm is a weighted spatial sum, so

    context = ReduceSum(x * mask, axes=[2,3], keepdims=True)

is algebraically identical and uses only Conv/Softmax/Mul/ReduceSum - no
MatMul, which matters because quantised MatMul is rejected on Hexagon v68
("incorrect Value 68, expected >= 73"). Verified against the trained forward
at 1.4e-06.

Upstream's OTHER two patches (StripPoolingAttention, MinimalCrossScale) were
checked and ARE faithful - adaptive_avg_pool2d(x,(H,1)) == x.mean(3,keepdim),
and adaptive_avg_pool2d(y, x_low.shape[2:]) == avg_pool2d(y,2,2) at 2x stride.
They exist only to give ONNX static shapes, so they are replicated as-is.

Usage: tools/zipdepth_gcb_export.py <out.onnx> <res> [--legacy-avgpool]
"""
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = r"C:\Users\kenchitaru-alex\Documents\GitHub\xreal-air2ultra-camera\ZipDepth"
sys.path.insert(0, REPO)
from zipdepth.model.architecture import create_model  # noqa: E402
from zipdepth.utils.model_utils import (  # noqa: E402
    fuse_remaining_conv_bn, strip_state_dict_prefixes)

out_path = sys.argv[1]
res = int(sys.argv[2])
legacy = "--legacy-avgpool" in sys.argv
CKPT = REPO + r"\checkpoints\zipdepth_base_npu.pth"

model = create_model(variant="base", global_mode="gcb", upsample_unfold=False)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
sd = strip_state_dict_prefixes(ckpt.get("model_state_dict", ckpt))
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
model = model.eval()
model.fuse_for_inference()
fuse_remaining_conv_bn(model)

H = W = res
s16 = (H // 16, W // 16)
n_gcb = 0
for m in model.modules():
    if type(m).__name__ == "GlobalContextBlock":
        if legacy:
            def _fwd(self_m, x, _h=s16[0], _w=s16[1]):
                ctx = F.avg_pool2d(x, kernel_size=(_h, _w))
                return x + self_m.transform(ctx)
        else:
            # Exact trained attention pooling, MatMul-free.
            def _fwd(self_m, x):
                B, C, HH, WW = x.shape
                mask = self_m.context_weight(x).view(B, 1, HH * WW)
                mask = F.softmax(mask, dim=-1).view(B, 1, HH, WW)
                ctx = (x * mask).sum(dim=(2, 3), keepdim=True)
                return x + self_m.transform(ctx)
        m.forward = types.MethodType(_fwd, m)
        n_gcb += 1
    elif type(m).__name__ == "StripPoolingAttention":
        def _fwd(self_m, x):
            B, C, HH, WW = x.shape
            gate = self_m.gate_conv(F.adaptive_avg_pool2d(x, (HH, 1))
                                    + F.adaptive_avg_pool2d(x, (1, WW)))
            return x * gate
        m.forward = types.MethodType(_fwd, m)

cs = model.encoder.cross_scale
def _cs_fwd(self_cs, x_high, x_low, _s=s16):
    lo = F.interpolate(self_cs.low_to_high(x_low), size=_s, mode="nearest")
    hi = F.avg_pool2d(self_cs.high_to_low(x_high), 2, 2)
    return x_high + lo * 0.3, x_low + hi * 0.3
cs.forward = types.MethodType(_cs_fwd, cs)

print(f"GlobalContextBlock: {n_gcb} patched -> "
      f"{'avg_pool2d (LEGACY, matches upstream)' if legacy else 'trained softmax attention (Mul+ReduceSum)'}")

dummy = torch.randn(1, 3, H, W)
with torch.no_grad():
    out = model(dummy)
print(f"pre-export sanity: output {tuple(out.shape)}")

torch.onnx.export(
    model, dummy, out_path, opset_version=17,
    input_names=["image"], output_names=["depth"],
    do_constant_folding=True, dynamic_axes=None,
)
print(f"WROTE {out_path}")

import collections  # noqa: E402
import onnx  # noqa: E402
g = onnx.load(out_path)
hist = collections.Counter(n.op_type for n in g.graph.node)
print("ops:", dict(sorted(hist.items(), key=lambda kv: -kv[1])))
for bad in ("MatMul", "Gemm", "Einsum"):
    if hist.get(bad):
        print(f"  WARNING: {hist[bad]}x {bad} present - quantised {bad} is rejected on v68")
print("Softmax:", hist.get("Softmax", 0), " ReduceSum:", hist.get("ReduceSum", 0))
