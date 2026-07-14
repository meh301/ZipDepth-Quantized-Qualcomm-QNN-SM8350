"""Wrap a ZipDepth HTP context binary as an EPContext ONNX for ORT's QNN EP.
Replicates the exact schema of the shipped zipdepth.onnx (verified attributes):
EPContext node, embed_mode=1, ep_sdk_version=2.48.0, main_context=1,
partition_name=QNNExecutionProvider_QNN_0, source=QNNExecutionProvider,
float32 I/O named image/depth, ir_version 9, opsets ("",17)+(com.microsoft,1).
Usage: python wrap_epctx.py <ctx.bin> <qdq_src.onnx> <out.onnx> <tag>"""
import sys, os
import onnx
from onnx import helper, TensorProto

ctx_path, src_path, out_path, tag = sys.argv[1:5]
ctx = open(ctx_path, "rb").read()
src = onnx.load(src_path)

in_name = src.graph.input[0].name
out_name = src.graph.output[0].name
in_shape = [d.dim_value for d in src.graph.input[0].type.tensor_type.shape.dim]
out_shape = [d.dim_value for d in src.graph.output[0].type.tensor_type.shape.dim]
# QDQ models may carry dynamic batch; pin to 1 like the shipped model.
in_shape = [(d if d > 0 else 1) for d in in_shape]
out_shape = [(d if d > 0 else 1) for d in out_shape]

node = helper.make_node(
    "EPContext", name="ZipDepth_QNN_ctx", inputs=[in_name], outputs=[out_name],
    domain="com.microsoft",
    embed_mode=1, ep_cache_context=ctx, ep_sdk_version="2.48.0", main_context=1,
    notes="ZipDepth AoT context %s, SM8350/V68, QAIRT 2.48" % tag,
    partition_name="QNNExecutionProvider_QNN_0", source="QNNExecutionProvider",
)
g = helper.make_graph(
    [node], "zipdepth_epctx_%s" % tag,
    [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, in_shape)],
    [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])
m.ir_version = 9
onnx.save(m, out_path)
onnx.checker.check_model(out_path)
print("WROTE %s %.2f MB (in=%s %s out=%s %s)" % (
    out_path, os.path.getsize(out_path) / 1e6, in_name, in_shape, out_name, out_shape))
