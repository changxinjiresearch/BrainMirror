#!/usr/bin/env python3
"""SCEPT U11 preregistered REAL pretrained MNIST-12 / MNIST test evaluation.

A scientific benchmark, not a Grad-CAM or clinical validation.
No training; no post-hoc metric changes; no patient MRI inference.
See docs/SCEPT_U11_PREREGISTRATION_V0_1.md BEFORE interpreting results.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # pure-numeric CI sanity tests; REAL model run explicitly requires ORT

MODEL_SHA = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"
MODEL_URL = ("https://huggingface.co/onnxmodelzoo/mnist-12/resolve/"
             "1cd752f4a3d818fa215aa621c78b8a10a4a1a3a5/mnist-12.onnx")
DATA_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
DATA_SHA = {
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
PRE_REGISTERED_SEEDS = (109, 211, 307)
PRE_REGISTERED_DOSES = (0.0, 0.5, 1.0)
PATCH_SIZE = 4
GRID = 7
N_PATCH = GRID * GRID
CTRL_MEAN_TOL = 0.20
CTRL_SD_TOL = 0.20
NOISE_SIGMA = .001
RISE_SAMPLES = 256
RISE_SEED = 8128
BOOTSTRAP_REPS = 2000
SIGN_REPS = 2000
STATS_SEED = 240216
ALPHA = 0.05
PRIMARY_PER_DIGIT = 10
PILOT_PER_DIGIT = 2
METHODS = ("scept", "occlusion", "rise", "random", "scept_no_nuisance")
REFS = ("occlusion", "rise", "random")
OUTCOMES = (("deletion_auc", -1), ("insertion_auc", 1))


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def download(url, dest, expected=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        result = digest(dest.read_bytes())
        if expected and result != expected:
            raise ValueError(f"HASH_MISMATCH {dest}: {result} != {expected}")
        return result
    req = urllib.request.Request(url, headers={"User-Agent": "SCEPT-U11-Research/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("RESOURCE_LIMIT unexpected asset > 20 MB")
    result = digest(data)
    if expected and result != expected:
        raise ValueError(f"HASH_MISMATCH {url}: {result} != {expected}")
    dest.write_bytes(data)
    return result


def read_idx(path, expected_magic):
    blob = gzip.decompress(Path(path).read_bytes())
    magic, n = struct.unpack_from(">II", blob, 0)
    if magic != expected_magic or n != 10000:
        raise ValueError(f"MNIST_IDX_SIGNATURE: {path}, magic={magic}, n={n}")
    if expected_magic == 2051:
        rows, cols = struct.unpack_from(">II", blob, 8)
        if (rows, cols) != (28, 28) or len(blob) != 16 + n * rows * cols:
            raise ValueError("MNIST_IMAGE_SHAPE_OR_SIZE")
        return np.frombuffer(blob, dtype=np.uint8, offset=16).reshape(n, 28, 28).copy()
    if len(blob) != 8 + n:
        raise ValueError("MNIST_LABEL_SHAPE_OR_SIZE")
    return np.frombuffer(blob, dtype=np.uint8, offset=8).copy()


def select_fixed_cases(labels, count):
    seen = [0] * 10
    ids = []
    for i, label in enumerate(labels):
        cl = int(label)
        if not 0 <= cl < 10:
            raise ValueError("INVALID_DIGIT_LABEL")
        if seen[cl] < count:
            ids.append(i)
            seen[cl] += 1
        if all(n == count for n in seen):
            break
    if seen != [count] * 10:
        raise ValueError("INCOMPLETE_PRE_REGISTERED_COHORT")
    return ids


def tiles():
    return tuple(np.array([(4 * r + y) * 28 + (4 * c + x)
            for y in range(4) for x in range(4)], dtype=np.intp)
            for r in range(7) for c in range(7))


TILES = tiles()


def stat(x, idx):
    v = x[idx]
    return float(v.mean()), float(v.std())


def choose_matched_controls(flat):
    desc = [stat(flat, ix) for ix in TILES]
    pairs = {}
    diagnostics = []
    for p in range(N_PATCH):
        candidates = []
        for q in range(N_PATCH):
            if p == q:
                continue
            dm = abs(desc[p][0] - desc[q][0])
            ds = abs(desc[p][1] - desc[q][1])
            if dm <= CTRL_MEAN_TOL and ds <= CTRL_SD_TOL:
                candidates.append((dm + ds, q, dm, ds))
        if candidates:
            _, q, dm, ds = min(candidates)
            pairs[p] = q
            diagnostics.append({"p": p, "q": q, "mean_diff": dm, "std_diff": ds})
        else:
            diagnostics.append({"p": p, "q": None, "reason": "CONTROL_UNMATCHED"})
    return pairs, diagnostics


def make_rng(seed):
    """Independent Python port of U10's JS mulberry32 + Box-Muller PRNG."""
    state = seed & 0xFFFFFFFF

    def rnd():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rnd


def u10_noise(flat, seed, sigma=NOISE_SIGMA):
    rnd = make_rng(seed)
    out = flat.copy()
    for i in range(out.size):
        u = max(rnd(), sys.float_info.min * sys.float_info.epsilon)
        v = rnd()
        noise = sigma * math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
        out[i] = np.float32(float(flat[i]) + noise)
    return out


def occlude(source, idx, dose=1):
    out = source.copy()
    out[idx] = np.asarray(out[idx] * (1.0 - dose), dtype=np.float32)
    return out


class Model:
    def __init__(self, file):
        if ort is None:
            raise RuntimeError('REAL_ONNX_RUNTIME_REQUIRED: install pinned onnxruntime')
        opt = ort.SessionOptions()
        opt.inter_op_num_threads = 1
        opt.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(file), sess_options=opt,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        if len(inp) != 1 or list(inp[0].shape) != [1, 1, 28, 28] or inp[0].type != "tensor(float)":
            raise ValueError("MODEL_INPUT_NOT_PRE_REGISTERED_MNIST12: " + str(inp))
        if len(outputs) != 1:
            raise ValueError("MODEL_OUTPUT_COUNT_UNSUPPORTED: " + str(outputs))
        self.input_name = inp[0].name
        self.output_name = outputs[0].name
        self.calls = 0

    def logits(self, x):
        if x.shape != (784,) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("INVALID_MODEL_INPUT")
        result = self.sess.run([self.output_name],
                               {self.input_name: x.reshape(1, 1, 28, 28)})[0]
        arr = np.asarray(result)
        if arr.size != 10 or arr.dtype not in (np.float32, np.float64):
            raise ValueError("MODEL_OUTPUT_NOT_TEN_CLASS_LOGITS")
        result = arr.reshape(-1).astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError("NONFINITE_REAL_INFERENCE")
        self.calls += 1
        return result

    def target(self, x, true_label):
        return float(self.logits(x)[true_label])


def softmax_true(scores, label):
    logits = scores - np.max(scores)
    prob = np.exp(logits)
    return float(prob[label] / prob.sum())


def ranked(score):
    if len(score) != N_PATCH:
        raise ValueError("SALIENCY_LENGTH_MISMATCH")
    return sorted(range(N_PATCH), key=lambda j: (-float(score[j]), j))


def auc(true_label, model, x, saliency):
    order = ranked(saliency)
    descending, ascending = [], []
    deleted = x.copy()
    inserted = np.zeros_like(x)
    descending.append(softmax_true(model.logits(deleted), true_label))
    ascending.append(softmax_true(model.logits(inserted), true_label))
    for j in order:
        deleted[TILES[j]] = 0
        inserted[TILES[j]] = x[TILES[j]]
        descending.append(softmax_true(model.logits(deleted), true_label))
        ascending.append(softmax_true(model.logits(inserted), true_label))
    grid = np.linspace(0, 1, N_PATCH + 1)
    return float(np.trapezoid(descending, grid)), float(np.trapezoid(ascending, grid))


def rise(model, x, cl, case_index):
    mask_rng = np.random.default_rng(RISE_SEED + case_index)
    score = np.zeros(N_PATCH, np.float64)
    for i in range(RISE_SAMPLES):
        m = (mask_rng.random(N_PATCH) < 0.5).astype(np.float32)
        grid_mask = np.repeat(np.repeat(m.reshape(7, 7), 4, axis=0), 4, axis=1).reshape(-1)
        prob = softmax_true(model.logits(np.asarray(x * grid_mask, dtype=np.float32)), cl)
        score += prob * m
    return (score / (RISE_SAMPLES * 0.5)).tolist()


def common_scept_map(model, x, cl, matched, noise_sigma):
    """U10 frozen definitions. B/C share z, A source is unchanged.
    This is a patchwise extension, not automatically equivalent to U10's
    pre-registered 7-gate study at patient/model level."""
    base = model.target(x, cl)
    per_patch = {p: [] for p in matched}
    all_measures = []
    noise_seeds = PRE_REGISTERED_SEEDS if noise_sigma else (PRE_REGISTERED_SEEDS[0],)
    for seed in noise_seeds:
        z = u10_noise(x, seed, sigma=noise_sigma) if noise_sigma else x.copy()
        cache = {}
        for dose in PRE_REGISTERED_DOSES:
            for p, q in matched.items():
                b = model.target(occlude(z, TILES[p], dose), cl)
                key = (q, dose)
                if key not in cache:
                    cache[key] = model.target(occlude(z, TILES[q], dose), cl)
                c = cache[key]
                specificity = abs(b - base) - abs(c - base)
                signed = b - c
                per_patch[p].append((seed, dose, specificity, signed))
    sal = [float("-inf")] * N_PATCH
    details = []
    for p, series in per_patch.items():
        full = [t for t in series if t[1] == 1.0]
        zcells = [t for t in series if t[1] == 0.0]
        sp = float(np.median([t[2] for t in full]))
        signed = np.array([t[3] for t in full], np.float64)
        drift = float(np.max(np.abs(signed - np.median(signed))))
        dose_violations = 0
        for seed in noise_seeds:
            doses = sorted([t for t in series if t[0] == seed], key=lambda v: v[1])
            magnitudes = [abs(t[3]) for t in doses]
            dose_violations += sum(magnitudes[i + 1] + 0.005 < magnitudes[i]
                                   for i in range(len(doses)-1))
        null = max(abs(t[3]) for t in zcells)
        # This benchmark deliberately exposes negative specificity and
        # candidate-level dose/null failures; it does NOT claim cohort gating.
        sal[p] = sp
        details.append({"p": p, "q": matched[p], "specificity": sp,
                        "full_signed_median": float(np.median(signed)),
                        "nuisance_spread": drift,
                        "dose_violations": dose_violations, "zero_dose_null": null,
                        "seed_count": len(noise_seeds)})
    return sal, details


def case_experiments(index, label, sample, model):
    x = np.asarray(sample.reshape(-1), np.float32) / np.float32(255)
    scores = model.logits(x)
    true_label = int(label)
    pred = int(np.argmax(scores))
    pairs, match_log = choose_matched_controls(x)
    matched_ratio = len(pairs) / N_PATCH
    sal, details = common_scept_map(model, x, true_label, pairs, NOISE_SIGMA)
    no_nuisance, _ = common_scept_map(model, x, true_label, pairs, 0.0)
    base = float(scores[true_label])
    occlusion = [base - model.target(occlude(x, tile), true_label) for tile in TILES]
    rise_map = rise(model, x, true_label, index)
    rand_rng = np.random.default_rng(7812 + index)
    random_scores = rand_rng.permutation(N_PATCH).astype(np.float64).tolist()
    maps = {"scept": sal, "occlusion": occlusion,
            "rise": rise_map, "random": random_scores,
            "scept_no_nuisance": no_nuisance}
    aucs = {}
    for method in METHODS:
        auc_d, auc_i = auc(true_label, model, x, maps[method])
        aucs[method] = {"deletion_auc": auc_d, "insertion_auc": auc_i}
    usable = (pred == true_label and matched_ratio >= 0.8)
    return {"case_id": index, "label": true_label, "prediction": pred,
            "correct": pred == true_label, "usable_primary": usable,
            "matched_n": len(pairs), "coverage": matched_ratio,
            "control_failures": N_PATCH-len(pairs),
            "reason": None if usable else ("misclassified" if pred != true_label
                    else "insufficient_matched_control_coverage"),
            "baseline_logit_true": base,
            "baseline_probability_true": softmax_true(scores, true_label),
            "scept_mean_specificity": float(np.mean([d["specificity"] for d in details]))
                if details else None,
            "scept_positive_fraction": (sum(d["specificity"] > 0 for d in details) / len(details))
                if details else None,
            "nuisance_spread_median": float(np.median([d["nuisance_spread"] for d in details]))
                if details else None,
            "dose_violations": int(sum(d["dose_violations"] for d in details)),
            "max_zero_dose": max((d["zero_dose_null"] for d in details), default=None),
            "aucs": aucs, "matching_log": match_log,
            "scept_measures": details,
            "tile_maps": {k: [None if not math.isfinite(v) else float(v) for v in vals]
                          for k, vals in maps.items()}}


def paired_stats(records):
    valid = [r for r in records if r["usable_primary"]]
    method_results = {}
    for method in METHODS:
        method_results[method] = {}
        for outcome, _ in OUTCOMES:
            method_results[method][outcome] = float(np.mean(
                [r["aucs"][method][outcome] for r in valid])) if valid else None
    comparisons = []
    if not valid:
        return {"methods": method_results, "comparisons": [],
                "n_primary": 0, "gate": "INCONCLUSIVE_NO_ELIGIBLE_IMAGES"}
    digits = sorted(set(r["label"] for r in valid))
    groups = {c: [r for r in valid if r["label"] == c] for c in digits}
    generator = np.random.default_rng(STATS_SEED)
    for baseline in REFS:
        for metric, direction in OUTCOMES:
            diff = np.array([(r["aucs"]["scept"][metric] -
                              r["aucs"][baseline][metric]) * direction for r in valid],
                            dtype=np.float64)
            draws = np.zeros(BOOTSTRAP_REPS, np.float64)
            for b in range(BOOTSTRAP_REPS):
                draws[b] = float(np.mean([(
                    rr["aucs"]["scept"][metric] -
                    rr["aucs"][baseline][metric]) * direction
                    for c in digits for rr in (
                        groups[c][j] for j in generator.integers(0, len(groups[c]),
                                                                 size=len(groups[c])))]))
            lower, upper = np.percentile(draws, [2.5, 97.5]).tolist()
            abs_obs = abs(float(diff.mean()))
            extreme = 0
            for _ in range(SIGN_REPS):
                signs = generator.choice(np.array([-1, 1]), size=diff.size)
                extreme += abs(float(np.mean(diff * signs))) >= abs_obs - 1e-15
            p = (extreme + 1) / (SIGN_REPS + 1)
            comparisons.append({"baseline": baseline, "outcome": metric,
                                "higher_is_better": True,
                                "signed_improvement_mean": float(diff.mean()),
                                "bootstrap_95ci": [float(lower), float(upper)],
                                "permutation_p_raw": float(p),
                                "n_images": len(valid)})
    sorted_indices = sorted(range(len(comparisons)),
                            key=lambda i: comparisons[i]["permutation_p_raw"])
    adjusted = [0] * len(comparisons)
    floor = 0.0
    for rank, ix in enumerate(sorted_indices):
        floor = max(floor, min(1.0, (len(comparisons) - rank) *
                               comparisons[ix]["permutation_p_raw"]))
        adjusted[ix] = floor
    for ix, item in enumerate(comparisons):
        item["holm_adjusted_p"] = float(adjusted[ix])
        item["statistical_signal"] = (adjusted[ix] < ALPHA and
                                      item["bootstrap_95ci"][0] > 0)
    return {"methods": method_results, "comparisons": comparisons,
            "n_primary": len(valid), "gate": "COMPUTED"}


def write_results(args, model_sha, asset_hashes, ids, imgs, labels):
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = Model(args.cache / "mnist-12.onnx")
    rows = []
    for no, idx in enumerate(ids):
        row = case_experiments(idx, labels[idx], imgs[idx], model)
        rows.append(row)
        print(f"[U11] {no+1}/{len(ids)} idx={idx} label={row['label']} "
              f"pred={row['prediction']} match={row['matched_n']}/{N_PATCH} "
              f"null={row['max_zero_dose']} calls={model.calls}", flush=True)
    analysis = paired_stats(rows)
    acc = sum(row["correct"] for row in rows) / len(rows)
    enough_coverage = sum(row["coverage"] >= .8 for row in rows) >= int(.8*len(rows))
    research_quality = enough_coverage and analysis["n_primary"] >= .8 * len(rows)
    report = {"schema": "scept.u11.research.benchmark.v0.1",
              "protocol": "docs/SCEPT_U11_PREREGISTRATION_V0_1.md",
              "model": {"source": MODEL_URL, "sha256": model_sha,
                        "input": model.input_name, "output": model.output_name,
                        "actual_runtime": "onnxruntime CPU"},
              "data": {"source": DATA_BASE, "source_hashes": asset_hashes,
                       "split": "official MNIST test IDX 10000 images", "selected_original_indices": ids,
                       "selection": f"first {PILOT_PER_DIGIT if args.mode == 'pilot' else PRIMARY_PER_DIGIT} examples per original digit",
                       "n": len(ids), "n_correct": sum(r["correct"] for r in rows),
                       "accuracy": acc},
              "design": {"fixed_grid": [7, 7], "patch": [4, 4],
                         "numeric_match_mean_tolerance": CTRL_MEAN_TOL,
                         "numeric_match_sd_tolerance": CTRL_SD_TOL,
                         "dose": list(PRE_REGISTERED_DOSES), "seeds": list(PRE_REGISTERED_SEEDS),
                         "noise_sigma": NOISE_SIGMA, "rise_samples": RISE_SAMPLES,
                         "statistical_replicates": [BOOTSTRAP_REPS, SIGN_REPS],
                         "stats_seed": STATS_SEED},
              "quality": {"average_coverage": float(np.mean([r["coverage"] for r in rows])),
                          "n_coverage_80pct": sum(r["coverage"] >= .8 for r in rows),
                          "n_not_evaluable": sum(not r["usable_primary"] for r in rows),
                          "all_zero_dose_verified": all((r["max_zero_dose"] is None or
                                                        r["max_zero_dose"] <= 1e-5)
                                                        for r in rows),
                          "dose_violations": int(sum(r["dose_violations"] for r in rows)),
                          "quality_gate": "PASS" if research_quality else "INCONCLUSIVE"},
              "statistics": analysis,
              "software": {"python": sys.version, "onnxruntime": ort.__version__,
                           "numpy": np.__version__, "system": platform.platform()},
              "runtime": {"actual_onnx_forward_calls": model.calls,
                          "wall_seconds": time.perf_counter() - started},
              "claims": {"engineering": "measured, not simulated",
                         "clinical": "NOT_ESTABLISHED", "causal": "NOT_ESTABLISHED",
                         "outperforms_all_xai": "NOT_ESTABLISHED",
                         "cross_model_generalization": "NOT_PERFORMED"},
              "limitations": ["single model/data family","CNN handwritten digit task",
                              "fixed zero-fill insertion/deletion OOD evaluator",
                              "patchwise SCEPT v1 extension not a validated saliency map",
                              "no Grad-CAM or Integrated Gradients baseline",
                              "no actual anatomical lesion annotation"]}
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False,
                                      indent=2, allow_nan=False) + "\n")
    (args.out / "cases.json").write_text(json.dumps(rows, ensure_ascii=False,
                                      indent=2, allow_nan=False) + "\n")
    with (args.out / "cases.csv").open("w", newline="") as f:
        w = csv.writer(f)
        header = ["case_id","digit","pred","correct","primary","coverage","matched_n",
                  "dose_violations","max_zero_dose"]
        header += [f"{m}_{metric}" for m in METHODS for metric,_ in OUTCOMES]
        w.writerow(header)
        for r in rows:
            w.writerow([r["case_id"],r["label"],r["prediction"],r["correct"],
                        r["usable_primary"],r["coverage"],r["matched_n"],
                        r["dose_violations"],r["max_zero_dose"]] +
                       [r["aucs"][m][metric] for m in METHODS for metric,_ in OUTCOMES])
    # Numerical invariants: repeated rerun of x-independent U10 controlled
    # contrast and null are checked at every case, not observational claims.
    if not report["quality"]["all_zero_dose_verified"]:
        raise AssertionError("FAILED_U10_ZERO_DOSE_NULL")
    print("U11_RESULT_SUMMARY_START", flush=True)
    print(json.dumps({k: report[k] for k in
            ("schema","model","data","quality","statistics","runtime","claims")},
            separators=(",",":"),allow_nan=False), flush=True)
    print("U11_RESULT_SUMMARY_END", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--out", type=Path, default=Path("results/u11/pilot"))
    parser.add_argument("--cache", type=Path, default=Path("/tmp/scept-u11-assets"))
    args = parser.parse_args()
    hashes = {"mnist-12.onnx": download(MODEL_URL,args.cache/"mnist-12.onnx",MODEL_SHA)}
    for name in FILES:
        hashes[name] = download(DATA_BASE+name,args.cache/name,DATA_SHA[name])
    imgs = read_idx(args.cache/FILES[0], 2051)
    labels = read_idx(args.cache/FILES[1], 2049)
    count = PILOT_PER_DIGIT if args.mode=="pilot" else PRIMARY_PER_DIGIT
    ids = select_fixed_cases(labels,count)
    report=write_results(args,hashes["mnist-12.onnx"],hashes,ids,imgs,labels)
    print(f"U11 FINISHED: {args.mode}; n={len(ids)}, acc={report['data']['accuracy']:.4f}, "
          f"n_primary={report['statistics']['n_primary']}, "
          f"quality={report['quality']['quality_gate']}",flush=True)


if __name__=="__main__":
    main()
