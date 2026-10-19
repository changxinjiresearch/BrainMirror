# SCEPT Research SDK A6 — External installation and M5 release qualification

**Status: candidate; M5 is blocked until independent acceptance evidence is attached.** This document defines testable engineering release criteria. A6 does **not** cure the U11 scientific efficacy failure or establish causal/clinical attribution.

## Completed in this candidate branch

- Source-installable Python SDK package with explicit A6 prerelease identifier and stable import surface; ZIP remains `scept.full-observation.v1`.
- Synthetic clean-room end-to-end example: one real PyTorch forward; frozen observed tensors; NAF; ZIP; hash-verified archive import; offline NAF recomputation; **separate real model reexecution**.
- Public API/version contract tests and CI test matrix for Python 3.10/3.11/3.12 (CPU only).
- Release acceptance sheet and third-party evidence template, with *pending* deliberately distinguished from *pass*.

## Clean operator procedure

Start from a fresh Python 3.10–3.12 virtual environment and a clean Git checkout at an immutable SHA:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install './scept_sdk[all,test]'
python scept_sdk/examples/a6_clean_room.py
python -m pytest scept_sdk/tests -q
python -m pip wheel --no-deps -w dist ./scept_sdk
python -m pip check
```

Use *synthetic* examples only; do not send patient MRI or downloaded observations to CI/GitHub. To verify a real observation offline use `scept-archive verify archive.zip` and `scept-archive recompute archive.zip`. Authenticity of a hostile unsigned archive is **not** established by SHA/CRC; verification is of internal consistency only. Real model replay also requires an independently supplied exact model and opt-in stored feed.

### Acceptance criteria

| Gate | Condition | Current attestation |
| --- | --- | --- |
| A6-G1 | Clean wheel/source install and CLI import across 3.10/3.11/3.12 | **PENDING CI run** |
| A6-G2 | Full A0–A5 real torch/ONNX/JS tests on final SHA | **PENDING CI run** |
| A6-G3 | Synthetic external-run example executes without repository patch | **PENDING CI run** |
| A6-G4 | Third party reruns on their own clean host and signs a reproducibility report | **NOT PERFORMED** |
| A6-G5 | Permission-holder selects and commits legal license, confirms third-party notices | **NOT APPROVED** |
| A6-G6 | Explicit compatibility policy, changelog, archive migration guarantees | **SPECIFIED HERE; NOT EXTERNALLY VALIDATED** |
| A6-G7 | M5 release decision references immutable commit, green CI and G4–G5 | **BLOCKED** |
| U11/M4 | Independently validated scientific superiority and falsification | **NOT ACHIEVED; SEPARATE GATE** |

### Stable contract and migration policy (alpha)

Package release candidate is `0.2.0a6`. **No stable 1.0 API or PyPI publication is claimed.** Public functions present in `scept.__all__` are the alpha API; changes to a function signature, output schema or NAF numerical semantics require a documented change and test. Persisted archives retain `scept.full-observation.v1`, source records retain their original `scept.python-capture.v1` or `scept.run-record.v1` schema, and no silent schema coercion is allowed. Future incompatible archives must use a new schema/version and an explicit conversion tool with source preservation and reproducibility tests. Scientific definitions and preregistered gates may not be silently modified by release engineering.

### Independent operator report

An *independent* tester (not the code author or this assistant) must record: date/host OS/CPU; Python, Torch, NumPy, ONNX, ONNX Runtime and Node versions; pristine repository commit SHA; wheel SHA-256; exact install/test commands, unabridged exit codes/logs; identity of any third-party example model and lawful data; test archive SHA and numerical replay tolerances; failures and deviations; operator identity/verification method. Store the signed report under `docs/verification/a6/` **without raw medical data or access tokens**. Empty template: [external report](SCEPT_SDK_A6_EXTERNAL_REPORT_TEMPLATE.md).

### Publishing and licensing

The repository historically does **not** assert a chosen license. **Do not invent permission, attach MIT/Apache, or push to PyPI** without repository-owner authorization. A source-installable prerelease and verified release candidate are not interchangeable with a licensed redistributable stable package. Record dependency notices, model/dataset permissions and clinical disclaimers before any public distribution.

**M5 verdict logic:** Only mark `PASS` with evidence for G1–G7 (including independent human execution and explicit licensing). Otherwise mark `BLOCKED` with reasons. No green CI implies medical efficacy.
