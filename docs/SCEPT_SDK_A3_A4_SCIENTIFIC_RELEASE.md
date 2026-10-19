# SCEPT Python SDK A3–A4: Scientific Development and Acceptance

## Scope and scientific claim

A3 computes real Python NAF from one genuine captured run. A4 supplies independent local protocol registration and true model-executed U10 A/B/C evidence dependence with preregistered gates.

These are numerical measurements, NOT proof of output importance, causal pathway, medical lesion evidence, clinical validity or performance superiority. A5 portable Python full archives and A6 external replication remain incomplete, and U11/M4 has NOT been achieved.

### A3: native internal-state calculations

For actual layer output H, profile defines native Token or channel coordinate u and its observed feature/spatial locations j. For n true values:

- mean(u) = sum_j H[u,j]/n
- RMS(u) = sqrt(sum_j H[u,j]^2/n)
- positiveFraction(u) = count(H[u,j] > 0)/n
- negativeFraction(u) = count(H[u,j] < 0)/n

PyTorch and ONNX outputs are obtained from one frozen real forward, never reconstructed from a visual heatmap. Per-unit channel/spatial pattern comes from original captured tensors. JavaScript V6 exact-token and native NCHW CNN formulas are compared against the SAME actual PyTorch tensor arrays in CI, with prespecified numeric tolerance. 3D CNN unit support in the Python SDK is an explicit extension not yet validated against a fully implemented JS 3D viewer.

For optional residual representation update, both pre- and post-unit tensors must be observed in the SAME forward, have identical float32 shape, a REAL directed path in the original frozen ONNX graph, and an explicitly registered same-residual-stream coordinate identity assertion. Otherwise return not_evaluable; matching shapes alone never qualify. On a supported native token u, RMSUpdate = sqrt(mean_j((post[u,j]-pre[u,j])^2)) and meanSignedUpdate=mean_j(post[u,j]-pre[u,j]). This is arithmetic change and not a mechanism of causation.

### A3: actual reference calibration

A3 also ports the production JS V6.3 reference and empirical midrank formulas. At least 5 real distinct input hashes with identical frozen model/preprocessing SHA, unit profile and actual observed shape are mandatory. The reference excludes its target input (leakage prevention), and within-volume ranks are not patient population percentiles. Explicit independent-cohort designation needs distinct caller-declared sample identities but does NOT independently prove patient independence. Exact JS midrank distributions are regression tested using 6 genuine PyTorch forwards. Fixed raw color magnitude is immutable across runs; adaptive per-run scaling cannot be mistaken for statistical calibration.

### A4: preregistered controlled test

Before executing inference, lock:
- exact model fingerprint, input name and numeric output tensor/index;
- target mask and disjoint equal-size matched control mask, externally documented source IDs, shapes, strictly sorted indices and SHA256 of little-endian Uint32 index bytes;
- ascending doses 0..1 with at least 3 levels, at least 3 independent nuisance UInt32 seeds, shared Gaussian law and fill value;
- mean/std matching tolerances, numeric output scale and ALL seven decision thresholds.

For each seed r and dose q:
- A = X (unchanged)
- Z_r = X + sigma * Gaussian(r), cast to actual typed input
- B = O_target(Z_r,q,fill), C = O_control(Z_r,q,fill), sharing exactly the SAME nuisance
- measure real frozen F(A), F(B), F(C) with three independent observed run IDs, numeric layer delta summaries and verifiable input hashes.

Set scalar a=s(F(A)), b=s(F(B)), c=s(F(C)), positive scale S:
- signed target delta = (b-a)/S
- signed control delta = (c-a)/S
- signed controlled contrast = (b-c)/S
- specificity score = (abs(b-a)-abs(c-a))/S

Seven original U10 gates: (1) target-vs-control specificity, (2) nonzero controlled effect, (3) replicate specificity, (4) nuisance spread limit, (5) signed direction agreement, (6) graded dose sensitivity, (7) zero-dose null. Missing, duplicate, inconsistent or unverified preregistered cells are not_evaluable; completed cells failing a numeric gate are criterion_not_met, all passing are criterion_met. Neither threshold tuning after observing outcomes nor interpretation as statistical p-values is authorized.

### Python usage

Package installs from local source as scept-research 0.2.0a4 (not published on PyPI).

Python example, assuming x is a real PyTorch model input:

    from scept import SCEPT
    observer = SCEPT(model, layers=["conv"], backend="torch", profile="cnn-nchw")
    result = observer.observe(x)
    naf = result.fingerprint()
    channel = result.pattern("conv", 0)["values"]

For an actual preregistered ONNX numerical controlled study:

    from scept import SCEPT, Mask, U10Plan, run_u10
    observer = SCEPT("frozen.onnx", layers=["real_layer"], backend="onnx")
    target = Mask.from_indices("input", tuple(x.shape), (0, 1), "target-source")
    control = Mask.from_indices("input", tuple(x.shape), (100, 101), "control-source")
    plan = U10Plan(input_name="input", target_tensor="actual_model_output",
                   target_index=0, target_mask=target, control_mask=control,
                   model_sha256=observer.model_sha256)
    report = run_u10(observer=observer, input_tensor=x, plan=plan)

The user must substitute actual existing layer names and scientifically suitable external ROI sources. Local ProtocolRegistry supports register(name, version, callable) and run(name, ...); arbitrary Python protocol plugins have local process privileges and ARE NOT untrusted-code sandboxes.

### Three scientific engineering problems and stage assignment

1. JS/Python numeric consistency: A3 compares real PyTorch tensor statistics and per-unit values against the production JS V6 core; A4 independently compares production U10 seed/noise and entire original JS seven-gate suite to Python outcomes from genuine ONNX A/B/C tasks. A5 extends this to full ZIP interoperability; A6 must verify broader real public model/runtime consistency.
2. Observation versus explanation: A3 states remain neutral measurements and unsupported graph updates fail closed; A4 U10 controlled tests have separate schemas and seven conditions. A5 reproducibility, A6 independent scientific tests and U11 baselines are still necessary.
3. Engineering maturity versus scientific validity: passing a genuine runtime test validates that code works on that task, not that an originality, statistical, causal or medical claim is true. M4 remains pending regardless of this SDK milestone.

### CI acceptance

Build/install source wheel, retain all A0–A2 tests, run real PyTorch ViT and CNN with JS numerical comparison, run genuine ONNX graph-connected update and reverse-path falsification, and run a complete 3 seeds x 3 doses x 3 real ONNX forwards study for positive and zero-effect negative models. Negative and corrupted masks/cell arrays must fail closed. Use original production JS U10 to cross-check exact noise and seven-gate results. No new paid GPU/cloud services.
