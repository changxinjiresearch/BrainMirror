# SCEPT U10 — Formal Explanation Mechanism v1.0

## Scientific status and scope

SCEPT-Evidential-Dependence-v1 is a **pre-registered paired controlled model-dependence test**, not a medically validated attribution map, inferred neuronal mechanism, or established causal estimator. It attaches to SCEPT Universal Core V1 without changing the generic U5 intervention executor, U6 scheduler, U7 evidence mechanism, or U8 data-driven views. Its four adapters are independent trusted SDK extensions in scept-web/plugins/u10_evidential_dependence.js. The browser study builder is scept-web/scept_u10_console.js.

### Input and frozen prerequisites

Let F_theta be a frozen verified ONNX graph; X its declared typed input and H_l(X) actual exposed intermediate output tensors. Researcher defines:

- F_theta SHA-256 and numeric target s(X): a *named real model output tensor* and a specific flattened numeric element j. This is not assumed to be a class probability.
- Explicit tensor-space **target** mask P and **matched control** mask Q, each with true input-name, shape, sorted unique flat indices, source ID, and computed SHA-256 checksum.
- Equal mask sizes and P∩Q=∅. Declare tolerances epsilon_mu, epsilon_sd for original pixel intensities of P and Q before ROI-specific intervention.
- Doses q in [0,1], sorted and unique, at least three including 0 and 1; independent nuisance PRNG seeds r (at least three); noise sigma in [0,0.1], fill v, output scale S>0, and decision thresholds locked before evaluating results.
- U6/U7 executes every (r,q) cell with three genuine ONNX forwards and records the frozen model/input SHA, runtime observations, seeds and matching metadata.

### Actual paired interventions

A = X is the **unchanged baseline**, because U5 guarantees plugin isolation and refuses modifications to the original source.

Let eta_r be a deterministic Gaussian numeric nuisance vector with user-registered sigma and seed. Define Z_r = X + eta_r, with the **same** Z_r used in both interventions B and C. The ROI operator O_R(Z,q,v) sets coordinate i in R to (1-q)*Z_i + q*v, leaving all other coordinates unchanged.

B_(r,q) = O_P(Z_r,q,v)
C_(r,q) = O_Q(Z_r,q,v).

For nonfloating feeds, source values are retained unchanged; all ROI interventions require a floating-point declared target feed. The noise is shared **between B and C only**, not A. Accordingly, the stability tested below is **the controlled B-minus-C contrast under shared nuisance**, NOT full-model robustness under different A baselines.

Before editing P or Q on Z_r, require |P|=|Q|, no overlapping flat indices, |mean(Z_r[P])-mean(Z_r[Q])| <= epsilon_mu and |sd(Z_r[P])-sd(Z_r[Q])| <= epsilon_sd. SHA, dimensions, sorted indices, numeric finite bounds and counts fail closed. These checks establish only *numerical* matching: neither spatial, biological nor lesion-anatomy matching is inferred.

### Mathematical explanation and observations

Let a=s(F_theta(A)), b_(r,q)=s(F_theta(B_(r,q))), c_(r,q)=s(F_theta(C_(r,q))). Compute:

DeltaP_(r,q) = (b_(r,q)-a)/S
DeltaQ_(r,q) = (c_(r,q)-a)/S
D_(r,q) = DeltaP_(r,q)-DeltaQ_(r,q) = (b_(r,q)-c_(r,q))/S
E_(r,q) = |D_(r,q)|

The **SCEPT specificity advantage**, the per-task numerical explanation score, is:

SCEPT_(r,q) = (|b_(r,q)-a| - |c_(r,q)-a|)/S.

The **controlled contrast** D is separate from the signed specificity margin SCEPT. E>0 only establishes different responses to the target and control perturbations; SCEPT>0 says the specified output's target-minus-A magnitude exceeds its matched control-minus-A magnitude. Their conjunction, subject to the gates below, is the *operational* interpretation criterion; neither alone proves pathology-specific evidence dependence.

For each *genuinely exported ONNX* layer output H_l, record mean(abs(H_l(B)-H_l(A))), mean(abs(H_l(C)-H_l(A))) and the numerical difference. They are **descriptive internal response measurements**, not pixel-level attributions, real neuron coordinates, or Grad-CAM-like saliency.

### Seven predefined decision gates

The independent-seed full-dose q=1 results define median controlled effect E_med=median_r E_(r,1), median specificity M_med=median_r SCEPT_(r,1), and p_pos=fraction of seeds where SCEPT_(r,1) >= tau_M.

Let D_med=median_r D_(r,1); H=max_r |D_(r,1)-D_med|; p_sign=fraction of seeds whose controlled signed effect has the expected sign. The sign may be preregistered as +1 or -1; exploratory null means use nonzero sign(D_med), and MUST NOT be described as prospectively chosen.

The suite passes **only if all**:
1. Matched-control specificity: M_med >= tau_M with a nonzero target response.
2. Controlled target evidence: E_med >= tau_E and E_med > 0 (zero effect cannot pass merely by stability).
3. Replicate specificity: p_pos >= pi_pos.
4. Nuisance-controlled effect stability: H <= tau_H.
5. Consistent direction: D_med != 0 and p_sign >= pi_sign.
6. Graded sensitivity: for every seed and ascending adjacent doses, E_(r,q_next) + tau_dose >= E_(r,q_prev).
7. Zero-dose null: max_r E_(r,0) <= tau_null. Because B_(r,0)=C_(r,0)=Z_r, B/C may each differ from A, while D_(r,0) must be zero.

Exactly the preregistered seed×dose matrix must complete. The adapter verifies real U7 triple-inference status, identical ROI checksums and nuisance sigma across tasks, unique cells, finite effects and numeric matching metadata. Incomplete, malformed, or duplicated evidence returns **not_evaluable**. Completed data failing any gate returns **criterion_not_met**. All seven passing returns **criterion_met** meaning *only this predefined computational test was satisfied*.

All thresholds must be fixed before measurement; the implementation does **not** calculate statistical p-values, confidence intervals, exchangeability/propensity scores, multiple testing corrections, patient-level generalization or a causal identification theorem. Doses sharing a seed are correlated. Report failures rather than threshold-tuning after looking at results.

### Negative controls and falsifiability

The mechanism is designed to reject:
- stable but zero-effect inputs;
- effect size no greater than the matched control;
- unstable sign or controlled magnitude under nuisance;
- violation of preregistered dose monotonicity;
- target/control intensity mismatch, overlap, altered mask checksum, wrong shape or nonfinite tensor data;
- absent actual control forward or exposed ONNX target;
- incomplete seed×dose grid or changed registered mask/perturbation law.

A meaningful scientific interpretation also requires external mask validity, anatomy-aware matching, evidence-preserving inpainting, OOD controls, independently held-out patients, multiple trained models, strong attribution baselines, statistical uncertainty and domain generalization. **None of those clinical/statistical claims is implied by passing U10.**

### Software interface and deployment

Independent trusted U9 SDK module exposes SCEPT_U10_MECHANISM.install({sdk,registry,core}) to register:
- scept.u10.paired-evidence-intervention (new A/B/C actual tensor operator)
- scept.u10.method.evidential-dependence (per-task real ONNX target/control explanation)
- scept.u10.eval.paired-evidence (per-task specificity check)
- scept.u10.eval.preregistered-gates (full seed×dose scientific decision procedure)

SCEPT_U10_MECHANISM.plan(...) constructs a complete U7 experiment manifest with model hash and external validated ROI SHA sources, 3 or more doses, 3 or more noise seeds, method, per-task and suite evaluations.

On the Universal page, the U10 panel builds a manifest from explicitly supplied ROI indices and actual loaded model signature. Researcher then follows existing U7 register → prepare → run; U8 automatically visualizes resulting genuine ONNX tensors, while U9 supports integrity-hashed audit and genuine replay. No new GPU, training, API key or paid hosting is required.

**Privacy:** U7 unredacted original manifest contains input ROI index arrays (potentially sensitive), and U10 does not automatically sanitize them. Use U9 redacted audit, review all metadata, and follow patient-data rules before publishing source masks. SHA-256 is integrity verification, not an author-authenticating signature.

**Version:** scept.u10.explanation.v1, frozen computational estimands. Later changes to nuisance assignment, scale, scores or gate definitions require new protocol version rather than silent revision.

### Explicit engineering-vs-science gate

The CI unit test executes a transparent numerical **mock** (not a real learned medical model) for preregistered gate logic, false-positive and replay behavior. The dedicated browser test executes an **untrained ONNX CNN** for real CPU/WASM A/B/C, model output/hidden observation, reproducibility and U7/U8 integration. Engineering tests do NOT demonstrate meaningful lesion interpretation, superiority over Grad-CAM, biological evidence or clinical validity.

U11 should supply carefully designed hypothesis tests, external data/model validation and comparison to saliency methods. **SCEPT U10 formally defines an explanatory hypothesis; it has not scientifically validated it.**
