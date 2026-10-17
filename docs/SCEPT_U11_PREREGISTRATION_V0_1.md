# SCEPT U11 — Pre-registered Real-Data Scientific Benchmark v0.1

**FROZEN BEFORE FIRST REAL-DATA RUN.** This protocol specifies analyses, sample selection and falsification *before* observing actual outcomes. Any deviation must be recorded as a numbered amendment. U11/M4 is **NOT achieved** by CI or installation alone.

## Research hypotheses

- H1 / fidelity: Under the same predetermined 4×4 input occlusion evaluation, SCEPT paired-control ROI ranking yields lower deletion AUC and higher insertion AUC than the random baseline; pairwise comparison with conventional unpaired occlusion and RISE is exploratory, not presumed positive.
- H2 / evidence vs stability: Stability of the numeric B–C contrast alone can occur when evidence strength/specificity is absent. The seven conjunctive U10 gates must reject this null.
- H3 / specificity: Swapping P and Q must reverse the sign of the signed specificity advantage, up to floating-point tolerance. Negative numeric specificity is recorded, not silently clamped.
- H4 / sensitivity: Monotone controlled response across q={0,0.5,1} is a testable hypothesis; any violation is recorded. Explicit numeric-matched control failures are counted and **never dropped silently**.

## Real independent sources

**Model:** [ONNX Model Zoo MNIST-12](https://huggingface.co/onnxmodelzoo/mnist-12/blob/main/mnist-12.onnx). Download pinned HF repo revision `1cd752f4a3d818fa215aa621c78b8a10a4a1a3a5`, model exact SHA256 `5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd`. Real pretrained CNN, float32 input NCHW [1,1,28,28], 10 real pre-softmax class scores; original source trained CNTK on MNIST (historical model card). If signature differs, stop rather than guess the class indices.

**Independent test data:** [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist) `https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz` + `.../t10k-labels-idx1-ubyte.gz`; read full 10,000 IDX images/labels, ensure correct magic and lengths, normalize pixel byte /255. Record download SHA256 and source URL. Never re-train or fine-tune on evaluation set.

## Pre-registered cohort and design

- Main: **first 10 images for each digit 0..9 in original chronological MNIST test IDX order** (100 fixed cases, not cherry-picked for successful classification). Pilot: first 2 per digit (20) using identical code; pilot is *not* included in final inferential results.
- Always report accuracy on all selected cases and per-digit counts. Primary deletion/insertion analysis uses **only correctly classified** cases (with all exclusions and reasons logged), while misclassified cases and their numeric explanations remain descriptive. The true-label logit is the frozen target for all cases; never switch to a convenient target after examining evidence.
- Divide 28×28 into **7×7 non-overlapping 4×4 regions**. Each has exactly 16 entries, 49 candidate target regions. Pair each P with a disjoint Q picked by pre-intervention mean/std intensity nearest distance (mean difference + std difference), deterministic stable index tie-break. Reject match if |mean(P)-mean(Q)|>0.20 OR |std(P)-std(Q)|>0.20; record absence as **not_evaluable**. Do not choose different thresholds based on observed outcomes.
- Fill value 0 (off-manifold warning); noise sigma 0.001; independent nuisance seeds [109,211,307] shared between B and C, **not A**; strengths [0,0.5,1]. These are repetitions within each case (not independent biological samples). Numerical scale S=1 logit unit.
- Per (P,Q,seed,dose), execute pretrained ONNX triple logic exactly as U10: a=f(A), b=f(B), c=f(C); specificity advantage = |b−a|−|c−a|; controlled contrast D=b−c; evidence strength |D|, no clamping. Study uses q=0 null, q=0.5 and q=1, tests dose monotonicity within seeds and max across seeds; any violation is a falsification, not a trigger to change the model/masks.
- Attribution map: median over nuisance seeds of full-dose SCEPT specificity advantage per 4×4 cell. Rejected control regions are marked unavailable; a per-case coverage fraction is disclosed. Rank eligible regions by score decreasing, break ties by original tile index. Negative scores remain negative. As an *ablation only*, rank by uncontrolled occlusion |f(B)-f(A)|.

## Reference explanation methods

1. **Occlusion sensitivity:** target-class logit decrease from full-dose zero-filled target patch, one forward per tile. No positive-clipping unless explicitly part of the sorting rule; signed decreases are ranked descending.
2. **RISE** randomized black-box input sampling: 256 binary 7×7 masks (independent RNG seed 8128), nearest-neighbor expanded to 28×28, Bernoulli p=0.5; score per grid cell = sum class probability f(x⊙M) M_j /(256×p). Real RISE is a stochastic estimator; sample budget and mask interpolation simplification disclosed.
3. **Random** ranking: independent per-case permutation of 49 cells using RNG seed 7812+case_index; no access to labels/model.
4. **SCEPT** as registered above, and an ablation with **no control** (plain occlusion) or no shared nuisance (sigma=0). The benchmark implements frozen v1 equations in independent Python and checks numerical parity against transparent test fixtures; this is not a new/unversioned SCEPT estimator.

**Grad-CAM and Integrated Gradients:** not included in this initial benchmark because the ONNX model gives no native autograd; actual pretrained gradient implementation and model-equivalence validation is a **mandatory extra validation gap before claims of broad superiority**. Never claim these baselines were run when they were not.

## Pre-registered outcomes

- Primary: **deletion AUC** (lower is more faithful) and **insertion AUC** (higher is more faithful), using target-class **softmax probability** evaluated at fractions k/49 for k=0..49. Mask insertion starts from all-zero image and reinstates ranked regions; deletion starts from original image and zeros top-ranked regions. Both use exact same forward operator for each method, with the associated potential circularity / OOD limitation. AUC uses trapezoidal mean over 50 points.
- Secondary: test-label accuracy, per-digit results, coverage (available P/Q), full-dose matched specificity sign proportions, 0-dose null, count/fraction of nonmonotonic dose steps, median perturbation-replicate score dispersion and Spearman agreement across repeat seeded runs.
- Controls: swap P and Q and require antisymmetric specificity up to 1e-5 logit (numerical invariant); stable zero-effect synthetic-null gate must fail independently in U10 unit suite; permute ROI/control assignments as specificity ablation (report null distribution); matched-control-unavailable rate explicitly.
- Do **not** use foreground pixel overlap as a biological explanation label; optional handwritten foreground fraction is descriptive only.

## Statistical analysis

Unit of inference = **unique original MNIST test image** (not 49 patches, 3 seeds, or doses). Paired differences (SCEPT minus baseline) in deletion and insertion AUC with seed 240216 deterministic **class-stratified bootstrap** (2000 resamples preserving per-digit case counts) 95% percentile confidence interval and **paired random sign permutation** (2000 draws). The p-value is a reproducible Monte Carlo two-sided estimate (plus-one correction), not an exact hypothesis test. Adjust the three baseline comparisons × two primary metrics using Holm correction. Report positive and negative effects, all adjusted p-values and CIs, all sample exclusions, and if score coverage is insufficient do not silently force ranking.
  
Declared minimum research-useful coverage: >=80% of all 49 candidate patches matched on >=80 of the 100 fixed cases; this is a *benchmark quality gate*, not a guarantee. No control-match and no successful enough cases = analysis inconclusive.

## Acceptance and claim separation

- **Engineering:** model+data hashes and signatures checked, real model forward accuracy, reproducible SCEPT and baseline maps, transparent matching failures, no fabricated observations, parity tests, negative controls, complete numeric result tables and generated plots, independent rerun with identical hashes.
- **Scientific:** any superiority claims require corresponding observed effect + uncertainty + corrected significance, no post-hoc thresholds, adequate coverage, honest failure reports. A single model and MNIST dataset **cannot establish cross-domain generalization or a clinically meaningful explanation**. Extra task families, trained medical models, validated masks, and Grad-CAM/IG gradient baselines are required before a broad **M4 Research V1** claim.
- Failures or null results do not invalidate software U10; they falsify narrow scientific hypotheses. The repository must preserve negative results unedited.

Resources: CPU-only GitHub Actions, no paid inference, no API key. Store completed JSON/CSV/figures alongside workflow artifacts; record runner, Python/ONNX Runtime versions and every file hash.