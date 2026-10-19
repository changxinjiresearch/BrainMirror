"""A6 independent-operator smoke example. Synthetic data only; no biomedical claims."""
from pathlib import Path
import json
import tempfile
import numpy as np
from scept import SCEPT, load_observation

def main():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the torch extra to run the PyTorch example") from exc
    torch.manual_seed(206)
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 3, 1), torch.nn.ReLU())
    observer = SCEPT(net, ["0", "1"], backend="torch", profile="cnn-nchw")
    x = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
    run = observer.observe(x, preprocess_id="synthetic-clean-room-a6")
    fingerprint = run.fingerprint()
    assert fingerprint["runId"] == run.run_id
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "single-observation.zip"
        run.export(str(path), include_inputs=True)
        imported = load_observation(str(path))
        for name in run.tensors:
            np.testing.assert_array_equal(imported.tensor(name), run.tensor(name))
        offline = imported.recompute_naf()
        assert offline["science"]["model_forward_reexecuted"] is False
        replay = imported.verify_model_replay(observer)
        assert replay["status"] == "numeric_reexecution_matched"
        assert replay["newIndependentRunId"] != run.run_id
        print(json.dumps({
            "status": "PASS_ENGINEERING_ONLY",
            "sourceRecordSchema": run.record["schema"],
            "observedTensors": len(run.tensors),
            "offlineRecomputeExecutedModel": False,
            "independentModelReexecution": True,
            "clinicalValidity": "NOT_ESTABLISHED",
            "scientificSuperiority": "NOT_ESTABLISHED"
        }, sort_keys=True))
if __name__ == "__main__":
    main()
