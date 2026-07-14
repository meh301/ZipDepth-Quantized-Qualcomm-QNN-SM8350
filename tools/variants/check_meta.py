"""Validate every compiled variant's context binary against the properties the
(working) shipped zipdepth binary has: float32 graph I/O, socModel 0."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

def describe(path, label):
    d = json.load(open(path, encoding="utf-8-sig"))
    info = d["info"]
    soc = info.get("socModel")
    ok = True
    for g in info.get("graphs", []):
        gi = g["info"]
        ins = [(t["info"]["name"], t["info"]["dataType"]) for t in gi.get("graphInputs", [])]
        outs = [(t["info"]["name"], t["info"]["dataType"]) for t in gi.get("graphOutputs", [])]
        io_ok = all(dt == "QNN_DATATYPE_FLOAT_32" for _, dt in ins + outs)
        soc_ok = soc == 0
        ok = ok and io_ok and soc_ok
        print(
            "%-12s soc=%s in=%s out=%s  %s"
            % (label, soc, ins, outs, "OK" if (io_ok and soc_ok) else "MISMATCH")
        )
    return ok

all_ok = True
print("-- shipped reference --")
describe(r"F:\xreal_depth\stereo_models\ctx\zipdepth_meta.json", "shipped")
print("-- variants --")
for meta in sorted(glob.glob(os.path.join(HERE, "ctx", "*", "meta.json"))):
    label = os.path.basename(os.path.dirname(meta))
    all_ok = describe(meta, label) and all_ok
print("ALL OK" if all_ok else "SOME MISMATCH")
