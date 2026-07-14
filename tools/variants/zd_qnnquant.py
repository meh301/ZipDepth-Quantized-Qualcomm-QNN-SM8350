r"""ZipDepth QDQ variants using the EXACT recipe that produced the shipped
(working) model — recovered from the original session's quantize_zipdepth.py:

  1. Fold every standalone BatchNormalization into Mul+Add (QNN HTP rejects
     BatchNorm with the signed per-channel Conv QDQ, error 3110).
  2. quant_pre_process (shape inference + optimization).
  3. get_qnn_qdq_config (the QNN-EP-specific QDQ builder) with MinMax
     calibration, per-channel signed INT8 weights, QUInt16 or QUInt8
     activations.
  4. quantize() — NOT plain quantize_static with hand-picked op lists.

Calibration mirrors the original: ZipDepth repo example photos, grayscale
replicated to 3 channels /255, augmented (crop/flip/brightness/gamma/noise)
plus synthetic range-pinning patterns.

Writes zipdepth_<label>_<res>.onnx into src/ for compile_variants.sh.
"""
import os
import glob

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import quantize, QuantType, CalibrationMethod
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
os.makedirs(SRC, exist_ok=True)
REPO = r"C:\Users\kenchitaru-alex\Documents\GitHub\xreal-air2ultra-camera"
SWEEP = r"F:\xreal_depth\zipdepth_sweep"
CKPT = os.path.join(REPO, r"ZipDepth\checkpoints")
IN_NAME = "image"

VARIANTS = [
    ("a8w8", QuantType.QUInt8, 384),
    ("a16w8", QuantType.QUInt16, 320),
    ("a8w8", QuantType.QUInt8, 320),
    ("a16w8", QuantType.QUInt16, 288),
    ("a8w8", QuantType.QUInt8, 288),
    ("a8w8", QuantType.QUInt8, 256),
]

rng = np.random.default_rng(0)


def fp32_path(res):
    if res == 384:
        return os.path.join(CKPT, "zipdepth_base_384x384.onnx")
    return os.path.join(SWEEP, "zipdepth_base_%d.onnx" % res)


def preprocess(bgr, res):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return np.repeat(gray[None, :, :], 3, axis=0)


def augment(bgr, res, n):
    out = []
    h0, w0 = bgr.shape[:2]
    for _ in range(n):
        img = bgr
        s = float(rng.uniform(0.5, 1.0))
        ch, cw = max(8, int(h0 * s)), max(8, int(w0 * s))
        y = int(rng.integers(0, h0 - ch + 1))
        x = int(rng.integers(0, w0 - cw + 1))
        img = img[y:y + ch, x:x + cw]
        if rng.random() < 0.5:
            img = img[:, ::-1]
        g = preprocess(img, res)
        g = g * float(rng.uniform(0.6, 1.4))
        g = np.clip(g, 0, 1) ** float(rng.uniform(0.7, 1.5))
        g = np.clip(g + rng.normal(0, 0.02, g.shape).astype(np.float32), 0, 1)
        out.append(g.astype(np.float32))
    return out


def synthetic(res):
    out = []
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(np.full((3, res, res), v, np.float32))
    for axis in (0, 1):
        ramp = np.linspace(0, 1, res, dtype=np.float32)
        g = np.tile(ramp, (res, 1)) if axis else np.tile(ramp[:, None], (1, res))
        out.append(np.repeat(g[None], 3, 0))
    chk = (np.indices((res, res)).sum(0) // 24 % 2).astype(np.float32)
    out.append(np.repeat(chk[None], 3, 0))
    return out


def build_calib(res):
    imgs = []
    pats = ["*_rgb.jpg", "im0.jpg", "im1.jpg", "*_rgb.png", "*.jpg", "*.png"]
    files = []
    for base in ("assets/examples", "assets/qualitative", "assets"):
        for p in pats:
            files += glob.glob(os.path.join(REPO, "ZipDepth", base, p))
    files = sorted(set(files))
    for f in files:
        bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        imgs.append(preprocess(bgr, res))
        imgs += augment(bgr, res, 48)
    imgs += synthetic(res)
    arr = np.stack(imgs, 0).reshape(-1, 3, res, res).astype(np.float32)
    print("calibration samples:", arr.shape, flush=True)
    return arr


class Reader(CalibrationDataReader):
    def __init__(self, arr):
        self.data = [{IN_NAME: arr[i:i + 1]} for i in range(arr.shape[0])]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        d = self.data[self.i]
        self.i += 1
        return d

    def rewind(self):
        self.i = 0


def fold_batchnorm(in_path, out_path):
    m = onnx.load(in_path)
    g = m.graph
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in g.initializer}
    out_nodes, n = [], 0
    for node in g.node:
        if node.op_type != "BatchNormalization":
            out_nodes.append(node)
            continue
        X = node.input[0]
        gamma, beta = inits[node.input[1]], inits[node.input[2]]
        mean, var = inits[node.input[3]], inits[node.input[4]]
        eps = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
        scale = (gamma / np.sqrt(var + eps)).astype(np.float32)
        shift = (beta - mean * scale).astype(np.float32)
        C = scale.shape[0]
        s_name, b_name = node.name + "_scale", node.name + "_shift"
        g.initializer.append(onnx.numpy_helper.from_array(scale.reshape(1, C, 1, 1), s_name))
        g.initializer.append(onnx.numpy_helper.from_array(shift.reshape(1, C, 1, 1), b_name))
        mul_out = node.name + "_mul_out"
        out_nodes.append(onnx.helper.make_node("Mul", [X, s_name], [mul_out], node.name + "_mul"))
        out_nodes.append(onnx.helper.make_node("Add", [mul_out, b_name], [node.output[0]], node.name + "_add"))
        n += 1
    del g.node[:]
    g.node.extend(out_nodes)
    onnx.save(m, out_path)
    print("folded BatchNorm -> Mul+Add:", n, flush=True)


calib_cache = {}
prep_cache = {}
for label, act, res in VARIANTS:
    fp32 = fp32_path(res)
    if not os.path.isfile(fp32):
        print("MISSING fp32:", fp32)
        continue
    if res not in prep_cache:
        folded = os.path.join(SRC, "zipdepth_folded_%d.onnx" % res)
        prep = os.path.join(SRC, "zipdepth_prep_%d.onnx" % res)
        fold_batchnorm(fp32, folded)
        quant_pre_process(folded, prep, skip_optimization=False, skip_symbolic_shape=False)
        prep_cache[res] = prep
    if res not in calib_cache:
        calib_cache[res] = build_calib(res)
    prep = prep_cache[res]

    out = os.path.join(SRC, "zipdepth_%s_%d.onnx" % (label, res))
    print("quantizing %s_%d ..." % (label, res), flush=True)
    cfg = get_qnn_qdq_config(
        prep, Reader(calib_cache[res]),
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=act,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    quantize(prep, out, cfg)
    onnx.checker.check_model(onnx.load(out))
    print("WROTE %s %.2f MB" % (out, os.path.getsize(out) / 1e6), flush=True)
print("DONE")
