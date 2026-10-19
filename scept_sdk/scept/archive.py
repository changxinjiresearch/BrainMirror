"""A5: bounded, integrity-verified portable scientific observation ZIP.

Reads BOTH browser V6.5.2 (scept.run-record.v1) and native Python
(scept.python-capture.v1) with unchanged record schemas. Verifies ALL binary
observations, hashes, shapes, provenance and NAF numeric states. An imported
archive is untrusted evidence, not proof its stated model actually ran.

Offline NAF replay NEVER invokes the model. An optional independently supplied
matching model/input can be explicitly re-executed, with numerical comparison.
No remote services, no ZIP extraction to disk, no arbitrary pickle/unpickle.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from .contracts import SCEPTError, fail, ensure_hash, tensor_array, tensor_sha, TYPE_MAP
from . import naf as naf_math

SCHEMA = "scept.full-observation.v1"
MAX_ARCHIVE_BYTES = 160 * 1024 * 1024
MAX_FILES = 100
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TENSOR_ELEMENTS = 8_000_000
SUPPORTED_RECORDS = ("scept.run-record.v1", "scept.python-capture.v1")
_VALID_KINDS = {"run_record","single_run_naf","empirical_reference",
                "observed_output_tensor","actual_model_input_tensor"}

def _json(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"), indent=2) + "\n").encode("utf-8")

def _sha(data: bytes) -> str: return sha256(data).hexdigest()

def _shape_dtype(shape: Any, dtype: Any) -> tuple[tuple[int, ...], np.dtype]:
    if dtype not in TYPE_MAP or not isinstance(shape, list) or not 1 <= len(shape) <= 8:
        fail("INVALID_TENSOR","Invalid dtype/shape in observed tensor")
    if any(type(d) is not int or d <= 0 for d in shape):
        fail("INVALID_TENSOR","Positive integer tensor axes only")
    total=math.prod(shape)
    if total>MAX_TENSOR_ELEMENTS:
        fail("RESOURCE_LIMIT","Archive tensor exceeds 8 million elements")
    return tuple(shape), np.dtype("uint8") if dtype=="bool" else TYPE_MAP[dtype]

def _tensor_raw(array: np.ndarray, dtype: str) -> bytes:
    native, typ = tensor_array(array)
    if typ != dtype: fail("TENSOR_TYPE_MISMATCH","Captured dtype changed")
    return native.tobytes(order="C")

def _read_json(raw: bytes, name: str) -> dict:
    if len(raw)>MAX_JSON_BYTES: fail("RESOURCE_LIMIT",f"JSON metadata too large: {name}")
    try: result=json.loads(raw)
    except (UnicodeError, ValueError) as e: fail("INVALID_JSON",f"{name}: {e}")
    if not isinstance(result,dict): fail("INVALID_JSON",f"{name} must contain an object")
    return result

def _numeric_same(a,b,*,atol=1e-8,rtol=1e-8):
    return (type(a) in (int,float) and type(b) in (int,float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a,b,abs_tol=atol,rel_tol=rtol))

def _match_naf(saved: dict, recalculated: dict) -> None:
    if saved.get("schema")!="scept.naf.v1" or saved.get("runId")!=recalculated.get("runId") or \
       saved.get("profile")!=recalculated.get("profile") or \
       saved.get("science",{}).get("status")!="single_run_observation_only":
        fail("NAF_MISMATCH","Archived NAF identity/status differs from raw captured observation")
    actual=recalculated["groups"]; original=saved.get("groups")
    if not isinstance(original,list) or len(original)!=len(actual):
        fail("NAF_MISMATCH","Missing/extra measured NAF groups")
    for x,y in zip(original,actual):
        if x.get("tensor")!=y["tensor"] or x.get("status")!=y["status"]:
            fail("NAF_MISMATCH","NAF tensor/status mismatch")
        if x["status"]!="measured": continue
        if x.get("role")!=y["role"] or x.get("shape")!=y["shape"]:
            fail("NAF_MISMATCH","NAF native axes/shape contradict actual raw tensors")
        rows=x.get("units")
        if not isinstance(rows,list) or len(rows)!=len(y["units"]):
            fail("NAF_MISMATCH","NAF unit count differs from raw tensor")
        for given,computed in zip(rows,y["units"]):
            if given.get("index")!=computed["index"] or given.get("count")!=computed["count"]:
                fail("NAF_MISMATCH","NAF index or actual coordinate count incorrect")
            for key in ("rms","mean","positiveFraction","negativeFraction"):
                if not _numeric_same(given.get(key),computed[key]):
                    fail("NAF_MISMATCH",f"NAF {key} contradicts actual stored float values")
    # Update values can be arithmetically checked if source tensors exist;
    # frozen source graph/proof CANNOT be reconstructed from a ZIP alone.
    for update in saved.get("updates",[]):
        if update.get("status")!="measured": continue
        start,end=update.get("inputTensor"),update.get("outputTensor")
        if start not in recalculated["_output_tensors"] or end not in recalculated["_output_tensors"]:
            fail("UPDATE_NOT_REPLAYABLE","Measured update source tensors not included in ZIP")
        src=recalculated["_output_tensors"][start][0]
        dst=recalculated["_output_tensors"][end][0]
        if src.shape!=dst.shape or src.ndim!=3 or src.shape[0]!=1 or src.dtype!=np.float32 or dst.dtype!=np.float32:
            fail("UPDATE_MISMATCH","Update tensors differ in shape/dtype")
        rows=update.get("units")
        if not isinstance(rows,list) or len(rows)!=src.shape[1]:
            fail("UPDATE_MISMATCH","Invalid update unit count")
        for i,row in enumerate(rows):
            d=dst[0,i].astype(np.float64)-src[0,i].astype(np.float64)
            if row.get("index")!=i or not _numeric_same(row.get("rmsUpdate"),
                float(np.sqrt(np.mean(d*d)))) or not _numeric_same(
                row.get("meanSignedUpdate"),float(np.mean(d))):
                fail("UPDATE_MISMATCH","Archived update numeric values contradict raw source tensors")

@dataclass(frozen=True)
class ArchiveObservation:
    """Only verified-file, UNTRUSTED external provenance; no new model run."""
    _manifest: dict
    _record: dict
    _naf: dict
    _outputs: dict[str,tuple[np.ndarray,str]]
    _inputs: dict[str,tuple[np.ndarray,str]]
    _reference: dict | None = None

    @property
    def manifest(self):return copy.deepcopy(self._manifest)
    @property
    def record(self):return copy.deepcopy(self._record)
    @property
    def saved_naf(self):return copy.deepcopy(self._naf)
    @property
    def run_id(self):return self._record["runId"]
    @property
    def tensors(self):return tuple(self._outputs)
    @property
    def input_names(self):return tuple(self._inputs)
    @property
    def reference(self):return copy.deepcopy(self._reference)

    def tensor(self,name: str) -> np.ndarray:
        if name not in self._outputs: fail("NOT_OBSERVED","No archived output: "+name)
        return self._outputs[name][0].copy()

    def input(self,name: str) -> np.ndarray:
        if name not in self._inputs: fail("INPUT_NOT_ARCHIVED","Input not explicitly included: "+name)
        return self._inputs[name][0].copy()

    def recompute_naf(self) -> dict:
        """Recompute native NAF numerical state from actual archived bytes.

        This is OFFLINE ARITHMETIC REPLAY, not re-execution of original model.
        """
        recomputed=_calculate_naf(self._record,self._naf,self._outputs)
        plain={k:v for k,v in recomputed.items() if not k.startswith("_")}
        plain["source"]="offline_recomputed_from_integrity_checked_archive"
        plain["science"]["model_forward_reexecuted"]=False
        plain["science"]["provenance_independently_authenticated"]=False
        return plain

    def verify_model_replay(self,observer: Any,*,atol:float=1e-6,rtol:float=1e-5) -> dict:
        """Explicitly run independently supplied frozen model; compare ALL outputs.

        Requires model inputs actually archived, matching model fingerprint,
        and matching the selected probes in independent observer. Fresh run
        is a separate inference and has a new run ID. Not same-run recording.
        """
        if not self._inputs:fail("INPUT_NOT_ARCHIVED","Model re-execution needs opt-in archived inputs")
        if not all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in (atol,rtol)):
            fail("INVALID_TOLERANCE","Predeclared finite nonnegative tolerances required")
        if not hasattr(observer,"model_sha256") or observer.model_sha256!=self._record["provenance"]["model"]["sha256"]:
            fail("MODEL_MISMATCH","Supplied independently executable frozen model does not match archived source fingerprint")
        input_names=set(self._inputs)
        if observer.backend=="onnx":
            if set(observer._adapter.inputs)!=input_names:
                fail("INPUT_MISMATCH","Real ONNX named model feeds differ from archive")
            values={name:a.copy() for name,(a,_) in self._inputs.items()}
        elif observer.backend=="torch":
            import torch
            if input_names!={"input"}:
                fail("INPUT_MISMATCH","Torch positional single-feed replay requires input")
            a=self._inputs["input"][0]
            parms=list(observer._adapter.model.parameters())
            device=parms[0].device if parms else "cpu"
            values=torch.from_numpy(a.copy()).to(device)
        else:
            fail("INVALID_BACKEND","Replay supports genuine torch/onnx only")
        fresh=observer.observe(values,preprocess_id="archived-input-explicit-new-forward",
                               model_id="separately-reexecuted-model")
        if set(fresh.tensors)!=set(self.tensors):
            fail("OUTPUT_MISMATCH","New runtime did not capture exactly all archived tensors")
        comparisons=[]
        for name in self.tensors:
            orig=self.tensor(name)
            new=fresh.tensor(name)
            if orig.shape!=new.shape:
                fail("OUTPUT_MISMATCH","Numerical replay shape differs")
            if orig.dtype.kind in "iub":
                ok=bool(np.array_equal(orig,new))
            else:
                ok=bool(np.allclose(orig,new,atol=atol,rtol=rtol,equal_nan=False))
            comparisons.append({"tensor":name,"matched":ok,
                                "maximumAbsoluteDifference":float(np.max(
                                  np.abs(orig.astype(np.float64)-new.astype(np.float64))))})
        return {"status":"numeric_reexecution_matched" if all(x["matched"] for x in comparisons)
                         else "numeric_reexecution_differed",
                "priorRunId":self.run_id,"newIndependentRunId":fresh.run_id,
                "modelHashMatched":True,"newModelForwardExecuted":True,
                "tolerance":{"atol":atol,"rtol":rtol},
                "perTensor":comparisons,
                "clinicalOrCausalMeaning":"NONE"}

def _calculate_naf(record:dict, saved:dict, outputs:dict) -> dict:
    """Reuse same A3 numeric code with an explicitly derived temporary view."""
    class OfflineProxy:
        def __init__(self):
            self.requested_probes=tuple(g["tensor"] for g in saved["groups"])
            self.tensors=tuple(outputs)
            self.record={"schema":"scept.python-capture.v1",
                 "runId":record["runId"],"provenance":record["provenance"],
                 "axisProfile":saved["profile"]}
        def tensor(self,name):return outputs[name][0].copy()
    calculation=naf_math.fingerprint(OfflineProxy())
    calculation["_output_tensors"]=outputs
    return calculation

def export_observation(capture, destination: str | Path, *,
                       include_inputs:bool=False,
                       reference:dict | None=None) -> dict:
    """Create cross-language layout-compatible ZIP; preserve native raw-record schema.

    The *original* native Python raw-capture schema is not rewritten into
    a browser V6 'observed activity' record. No fake graph annotation.
    """
    if type(include_inputs) is not bool:fail("INVALID_OPTION","Explicit Boolean consent required")
    record=capture.record;naf=capture.fingerprint()
    if record.get("schema")!="scept.python-capture.v1" or naf.get("runId")!=record.get("runId"):
        fail("INVALID_CAPTURE","A3 genuine native Python numeric capture required")
    output_records=record["outputs"]
    if len(output_records)!=len(capture.tensors) or {o["name"] for o in output_records}!=set(capture.tensors):
        fail("INCOMPLETE_CAPTURE","Must archive every originally recorded output")
    entries=[];blobs=[];total=0
    def add(path:str,raw:bytes,kind:str,**metadata):
        nonlocal total
        if len(blobs)+2>MAX_FILES:fail("RESOURCE_LIMIT","Archive has too many data files")
        total+=len(raw)
        if total>MAX_ARCHIVE_BYTES:fail("RESOURCE_LIMIT","Archive data exceeds 160 MiB")
        blobs.append((path,raw))
        entries.append({"path":path,"bytes":len(raw),"sha256":_sha(raw),"kind":kind,**metadata})
    add("record.json",_json(record),"run_record")
    add("naf.json",_json(naf),"single_run_naf")
    if reference is not None:
        if reference.get("schema")!="scept.naf.reference.v1" or \
           reference.get("modelSha256")!=record["provenance"]["model"]["sha256"] or \
           reference.get("preprocessSha256")!=record["provenance"]["preprocessing"]["sha256"] or \
           reference.get("profile")!=naf["profile"]:
            fail("INVALID_REFERENCE","Reference identity mismatch")
        add("reference.json",_json(reference),"empirical_reference")
    for index,o in enumerate(output_records):
        name=o["name"];array=capture.tensor(name);dt=o["type"]
        data,realtype=tensor_array(array)
        if dt!=realtype or o["shape"]!=list(data.shape) or o["sha256"]!=tensor_sha(data,dt):
            fail("TENSOR_HASH_MISMATCH","Observed source differs from recorded output SHA: "+name)
        add(f"tensors/outputs/{index:03}.bin",_tensor_raw(data,dt),
            "observed_output_tensor",tensorName=name,dtype=dt,shape=list(data.shape),
            tensorSha256=o["sha256"],byteOrder="little-endian-runtime-typed-array")
    if include_inputs:
        sigs=record["signature"]["inputs"]
        if {s["name"] for s in sigs}!=set(capture.input_names):
            fail("INCOMPLETE_CAPTURE","Cannot include input not truly retained")
        for index,i in enumerate(sigs):
            arr=capture.input(i["name"])
            dat,dt=tensor_array(arr)
            if i["type"]!=dt or list(dat.shape)!=i["shape"] or \
               tensor_sha(dat,dt)!=record["provenance"]["inputs"][i["name"]]:
                fail("INPUT_HASH_MISMATCH","Raw model input changed since inference")
            add(f"tensors/inputs/{index:03}.bin",_tensor_raw(dat,dt),
                "actual_model_input_tensor",tensorName=i["name"],dtype=dt,
                shape=list(dat.shape),tensorSha256=record["provenance"]["inputs"][i["name"]],
                byteOrder="little-endian-runtime-typed-array",userConsent=True)
    missing=[o for o in record.get("availability",[]) if o.get("status")!="measured"]
    manifest={"schema":SCHEMA,"version":"6.5.2-python-a5","runId":record["runId"],
       "createdAt":record["createdAt"],"sourceRecordSchema":record["schema"],
       "scientificStatus":"OBSERVATION_ONLY_NOT_CAUSAL_OR_CLINICALLY_VALIDATED",
       "captureScope":{"allDeclaredRuntimeOutputs":True,
          "storedOutputTensors":len(output_records),"rawOutputBytes":sum(
             capture.tensor(o["name"]).nbytes for o in output_records),
          "inputsIncluded":include_inputs,"modelWeightsIncluded":False,
          "originalNiftiSourceIncluded":False,"otherModelInternalsNotInstrumented":True,
          "unavailableProbes":missing,"previousRunRawDataIncluded":False},
       "notes":["Full original measured raw output tensors; uninstrumented layers NOT included.",
                "A native Python raw-capture record and A3 computed NAF are separate typed artifacts.",
                "Input data optional; no model weights or executable preprocessing source.",
                "ZIP proves internal file consistency; author/model identity remains untrusted."],
       "files":copy.deepcopy(entries)}
    data=_json(manifest)
    if total+len(data)+4096+sum(len(n)+100 for n,_ in blobs)>MAX_ARCHIVE_BYTES:
        fail("RESOURCE_LIMIT","Archive exceeds bounded 160 MiB")
    target=Path(destination)
    if not target.name or target.is_dir():
        fail("INVALID_PATH","Explicit ZIP target file required")
    # Never load dynamic code from external ZIP; normal pure-stdlib ZIP/STORE.
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,"w",compression=zipfile.ZIP_STORED,
                         allowZip64=False) as z:
        for name,raw in blobs:z.writestr(name,raw)
        z.writestr("manifest.json",data)
    if memory.tell()>MAX_ARCHIVE_BYTES:
        fail("RESOURCE_LIMIT","Final ZIP too large")
    # Validate BEFORE writing anything to user's chosen destination.
    output=memory.getvalue()
    _load_zip(output)
    target.write_bytes(output)
    return {"path":str(target),"bytes":len(output),"runId":record["runId"],
            "outputTensors":len(output_records),
            "inputsIncluded":include_inputs,"fileCount":len(blobs)+1,
            "status":"archive_written_and_verified_not_scientifically_authenticated"}

def load_observation(source: str | Path | bytes) -> ArchiveObservation:
    """Safe, full-integrity read of Python or website V6.5 full-observation ZIP."""
    if isinstance(source,bytes):
        if len(source)>MAX_ARCHIVE_BYTES:fail("RESOURCE_LIMIT","Archive exceeds 160 MiB")
        return _load_zip(source)
    path=Path(source)
    if path.stat().st_size>MAX_ARCHIVE_BYTES:fail("RESOURCE_LIMIT","Archive exceeds 160 MiB")
    return _load_zip(path.read_bytes())

def _load_zip(data: bytes) -> ArchiveObservation:
    try:
        z=zipfile.ZipFile(io.BytesIO(data),"r",allowZip64=False)
    except (zipfile.BadZipFile,ValueError,OSError) as e:
        fail("INVALID_ZIP",str(e))
    with z:
        infos=z.infolist()
        if not 3<=len(infos)<=MAX_FILES:
            fail("RESOURCE_LIMIT","Invalid archive file count")
        seen=set()
        total=0
        for inf in infos:
            n=inf.filename
            if n in seen or not n or len(n)>161 or n.startswith("/") or ".." in n or \
               "\\" in n or n.endswith("/") or not all(ch.isascii() and
                 (ch.isalnum() or ch in "/._-") for ch in n):
                fail("INVALID_PATH","Duplicate/traversal/unsafe ZIP member")
            seen.add(n)
            if inf.compress_type!=zipfile.ZIP_STORED or inf.flag_bits&1:
                fail("INVALID_ZIP","Compressed or encrypted archive rejected")
            if inf.file_size>MAX_ARCHIVE_BYTES or inf.compress_size!=inf.file_size:
                fail("RESOURCE_LIMIT","ZIP member size invalid")
            if inf.external_attr>>16 & 0o170000 == 0o120000:
                fail("INVALID_PATH","No symlink ZIP members")
            total+=inf.file_size
            if total>MAX_ARCHIVE_BYTES:fail("RESOURCE_LIMIT","Uncompressed payload too large")
        if not {"manifest.json","record.json","naf.json"}.issubset(seen):
            fail("INVALID_ZIP","Missing mandatory science/provenance metadata")
        manifest=_read_json(z.read("manifest.json"),"manifest")
        record=_read_json(z.read("record.json"),"record")
        naf=_read_json(z.read("naf.json"),"naf")
        if manifest.get("schema")!=SCHEMA or record.get("schema") not in SUPPORTED_RECORDS or \
           naf.get("schema")!="scept.naf.v1" or manifest.get("runId")!=record.get("runId") or \
           naf.get("runId")!=record.get("runId"):
            fail("INVALID_SCHEMA","Independent native/browser science schema or run ID invalid")
        if manifest.get("sourceRecordSchema",record["schema"])!=record["schema"]:
            fail("INVALID_SCHEMA","Source record schema declaration mismatch")
        if naf.get("provenance")!=record.get("provenance") or \
           naf.get("science",{}).get("status")!="single_run_observation_only":
            fail("INVALID_PROVENANCE","NAF scientific/source provenance mismatch")
        ensure_hash(record["provenance"]["model"]["sha256"],"model")
        if not record["provenance"].get("inputs") or not all(
            ensure_hash(x,"input") for x in record["provenance"]["inputs"].values()):
            fail("INVALID_PROVENANCE","Named original input SHA missing")
        entries=manifest.get("files")
        if not isinstance(entries,list) or not entries:
            fail("INVALID_SCHEMA","Missing ordered file entries")
        expected={entry.get("path") for entry in entries if isinstance(entry,dict)}
        if len(expected)!=len(entries) or expected|{"manifest.json"}!=seen:
            fail("INCOMPLETE_ARCHIVE","Missing/extra/duplicate data entries")
        for entry in entries:
            n=entry["path"]
            if entry.get("kind") not in _VALID_KINDS or n=="manifest.json":
                fail("INVALID_SCHEMA","Unrecognized file kind/name")
            raw=z.read(n)   # zipfile verifies CRC32 independently
            if len(raw)!=entry.get("bytes") or _sha(raw)!=entry.get("sha256"):
                fail("HASH_MISMATCH","File SHA/byte length does not match manifest: "+n)
        if {e["path"] for e in entries if e["kind"]=="run_record"}!={"record.json"} or \
           {e["path"] for e in entries if e["kind"]=="single_run_naf"}!={"naf.json"}:
            fail("INVALID_SCHEMA","Missing genuine record/NAF file kind")
        refs=[e for e in entries if e["kind"]=="empirical_reference"]
        if len(refs)>1 or (refs and refs[0]["path"]!="reference.json"):
            fail("INVALID_REFERENCE","Invalid empirical reference filename")
        ref=_read_json(z.read("reference.json"),"reference") if refs else None
        if ref is not None and (ref.get("schema")!="scept.naf.reference.v1" or
           ref.get("modelSha256")!=record["provenance"]["model"]["sha256"] or
           ref.get("preprocessSha256")!=record["provenance"]["preprocessing"]["sha256"] or
           ref.get("profile")!=naf["profile"]):
            fail("INVALID_REFERENCE","Empirical reference not model/input/preprocessing matched")
        decl=record.get("outputs")
        if not isinstance(decl,list) or not decl or len({x["name"] for x in decl})!=len(decl):
            fail("INVALID_RECORD","Output provenance absent/duplicated")
        source_outputs={o["name"]:o for o in decl}
        outs=[e for e in entries if e["kind"]=="observed_output_tensor"]
        if len(outs)!=len(decl) or len({e["tensorName"] for e in outs})!=len(decl) or \
           {e["tensorName"] for e in outs}!=set(source_outputs):
            fail("INCOMPLETE_ARCHIVE","Not all originally declared output tensors were stored")
        inputs_decl=record.get("signature",{}).get("inputs",[])
        if not isinstance(inputs_decl,list) or not inputs_decl:
            fail("INVALID_RECORD","Missing named input signatures")
        input_sig={x["name"]:x for x in inputs_decl}
        in_entries=[e for e in entries if e["kind"]=="actual_model_input_tensor"]
        if bool(in_entries)!=bool(manifest.get("captureScope",{}).get("inputsIncluded")):
            fail("INPUT_CONSENT_MISMATCH","Input capture scope contradicts stored payloads")
        if in_entries and (len(in_entries)!=len(inputs_decl) or
                           {e["tensorName"] for e in in_entries}!=set(input_sig) or
                           any(e.get("userConsent") is not True for e in in_entries)):
            fail("INPUT_CONSENT_MISMATCH","Partial/unconsented original input tensors")
        actual_outputs={};actual_inputs={}
        for kind,items,metadata,destination in [
            ("observed_output_tensor",outs,source_outputs,actual_outputs),
            ("actual_model_input_tensor",in_entries,input_sig,actual_inputs)]:
            for entry in items:
                name=entry["tensorName"];meta=metadata[name]
                shape,dtype=_shape_dtype(entry.get("shape"),entry.get("dtype"))
                raw=z.read(entry["path"])
                if entry.get("byteOrder")!="little-endian-runtime-typed-array" or \
                   len(raw)!=math.prod(shape)*dtype.itemsize or \
                   entry.get("dtype")!=meta["type"] or \
                   list(shape)!=meta["shape"]:
                    fail("TENSOR_MISMATCH","Shape/dtype/byte-order mismatch: "+name)
                array=np.frombuffer(raw,dtype=dtype.newbyteorder("<")).copy().reshape(shape)
                actual_sha=tensor_sha(array,entry["dtype"])
                expected_hash=meta["sha256"] if kind=="observed_output_tensor" else \
                    record["provenance"]["inputs"][name]
                if actual_sha!=ensure_hash(entry.get("tensorSha256"),"tensor") or \
                   actual_sha!=ensure_hash(expected_hash,"record"):
                    fail("TENSOR_HASH_MISMATCH","Raw bytes contradict record: "+name)
                if kind=="observed_output_tensor" and (entry.get("kind") not in _VALID_KIN or
                    meta.get("elements")!=array.size or meta.get("status")!="measured"):
                    fail("INVALID_RECORD","Runtime recorded output dimension/status invalid")
                destination[name]=(array,entry["dtype"])
        scope=manifest.get("captureScope",{})
        if scope.get("allDeclaredRuntimeOutputs") is not True or scope.get("storedOutputTensors")!=len(decl) or \
           scope.get("rawOutputBytes")!=sum(a.nbytes for a,_ in actual_outputs.values()) or \
           scope.get("modelWeightsIncluded") is not False or \
           scope.get("originalNiftiSourceIncluded") is not False or \
           scope.get("previousRunRawDataIncluded") is not False:
            fail("INCOMPLETE_ARCHIVE","Capture completeness or explicit exclusions invalid")
        if record["schema"]=="scept.python-capture.v1":
            expected_probes=[o["tensor"] for o in record.get("observations",[])]
        else:
            expected_probes=[o["tensor"] for o in record.get("observations",[])]
        if expected_probes!=[g.get("tensor") for g in naf.get("groups",[])]:
            fail("NAF_MISMATCH","NAF does not represent exactly declared requested probes")
        calc=_calculate_naf(record,naf,actual_outputs)
        _match_naf(naf,calc)
        return ArchiveObservation(manifest,record,naf,actual_outputs,actual_inputs,ref)
