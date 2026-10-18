#!/usr/bin/env python3
"""External/independent stdlib validator for SCEPT ZIP/STORE output.

Validates actual file-level hashes, ZIP CRC, listed tensor shape and SHA-256
type+shape+byte identity. Does NOT establish that a foreign manifest came from
the claimed weights, real patient source, or causal interpretation.
"""
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

def verify(path: str):
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("Missing manifest / duplicate entries")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["schema"] != "scept.full-observation.v1":
            raise ValueError("Wrong archive schema")
        entries = manifest["files"]
        expected = {entry["path"] for entry in entries} | {"manifest.json"}
        if set(names) != expected:
            raise ValueError("Unlisted files / missing recorded files")
        outputs = [x for x in entries if x["kind"] == "observed_output_tensor"]
        if len(outputs) != manifest["captureScope"]["storedOutputTensors"]:
            raise ValueError("Missing declared output tensors")
        record = json.loads(archive.read("record.json"))
        naf = json.loads(archive.read("naf.json"))
        if record["runId"] != naf["runId"] or record["runId"] != manifest["runId"]:
            raise ValueError("Run ID mismatch")
        declared = {o["name"]: o for o in record["outputs"]}
        for x in entries:
            raw = archive.read(x["path"])
            if len(raw) != x["bytes"] or hashlib.sha256(raw).hexdigest() != x["sha256"]:
                raise ValueError("File byte count/SHA-256 mismatch: " + x["path"])
            if x["kind"] not in ("observed_output_tensor", "actual_model_input_tensor"):
                continue
            shape, dtype = x["shape"], x["dtype"]
            count = 1
            for n in shape:
                if not isinstance(n, int) or n <= 0:
                    raise ValueError("Invalid dimensions")
                count *= n
            widths = {"float32": 4, "float64": 8, "int32": 4, "int64": 8, "uint8": 1, "bool": 1}
            if len(raw) != count * widths[dtype]:
                raise ValueError("Tensor shape/dtype size mismatch")
            header = json.dumps({"shape": shape, "type": dtype},
                separators=(",", ":"), sort_keys=True).encode()
            tensor_digest = hashlib.sha256(header + b"\0" + raw).hexdigest()
            if tensor_digest != x["tensorSha256"]:
                raise ValueError("SCEPT canonical tensor SHA mismatch")
            if x["kind"] == "observed_output_tensor":
                if x["tensorName"] not in declared or declared[x["tensorName"]]["sha256"] != tensor_digest:
                    raise ValueError("Output mismatches Run Record")
            else:
                if record["provenance"]["inputs"][x["tensorName"]] != tensor_digest:
                    raise ValueError("Input mismatches Run Record")
        print("PASS: independent SCEPT ZIP CRC, exact output payloads, JSON manifest/file SHA and dtype/shape canonical hashes; tensors =", len(outputs))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python verify_full_archive.py <scept-full-observation.zip>")
    verify(sys.argv[1])
