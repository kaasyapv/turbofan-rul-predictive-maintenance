"""Export the trained LSTMs to ONNX and check ONNX Runtime against PyTorch on the test set."""
import argparse
import numpy as np
import onnxruntime as ort
import torch
from . import data, model
from .metrics import clip, rmse


def export(name: str):
    _, sets, _ = data.build(name)
    x, true = sets["te"]["x"], sets["te"]["y"]
    ref, got = [], []
    for seed in range(3):
        net = model.Net(x.shape[2])
        net.load_state_dict(torch.load(f"models/{name}/lstm_{seed}.pt"))
        path = f"models/{name}/lstm_{seed}.onnx"
        # dynamo=False because the new exporter does not handle LSTM yet
        torch.onnx.export(net.eval(), torch.zeros(1, x.shape[1], x.shape[2]), path, opset_version=17,
                          input_names=["x"], output_names=["rul"], dynamo=False,
                          dynamic_axes={"x": {0: "batch"}, "rul": {0: "batch"}})
        ref.append(clip(model.predict(net, x)))
        got.append(clip(ort.InferenceSession(path).run(None, {"x": x.astype(np.float32)})[0] * 125))
    for tag, a, b in (("seed 0", ref[0], got[0]), ("ensemble", np.mean(ref, 0), np.mean(got, 0))):
        print(f"{name} {tag}: torch rmse {rmse(a, true)!r} onnx rmse {rmse(b, true)!r} "
              f"max pred diff {np.abs(a - b).max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subsets", nargs="*", default=["FD001", "FD002", "FD003", "FD004"])
    for name in ap.parse_args().subsets:
        export(name)
