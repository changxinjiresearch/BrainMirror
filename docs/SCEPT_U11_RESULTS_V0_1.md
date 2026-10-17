# SCEPT U11 — Empirical results, preregistered benchmark v0.1

**Scientific verdict: M4 NOT YET ESTABLISHED.** This is an actual test of a pretrained model and authentic independent data, not synthetic inference. The main SCEPT U10 patchwise ranking is significantly worse than two existing explanatory baselines on these deletion/insertion metrics, despite clearly exceeding random.

## Frozen study sources

- [Original protocol, committed before results](SCEPT_U11_PREREGISTRATION_V0_1.md)
- [Actual 100-case reproducibility workflow](https://github.com/changxinjiresearch/BrainMirror/actions/runs/39072314873)
- [Pretrained ONNX Model Zoo MNIST-12](https://huggingface.co/onnxmodelzoo/mnist-12) SHA256: 5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd
- [CVDF official MNIST test IDX data](https://github.com/cvdfoundation/mnist): 10,000 image/label pairs. Image gzip SHA256 8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6; label gzip SHA256 f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6.
- First 10 images per digit, original test-set order, exactly 100 fixed images. Classifier accurate on 98 of 100 (98%). **All 98 correctly classified cases** enter preregistered primary faithfulness analysis. Excluded two classification failures remain in case records.
- Fixed 49 nonoverlapping 4x4 patches; disjoint P/Q matched by pre-nuisance mean/std within 0.20. Shared B/C Gaussian nuisance sigma 0.001, seeds 109/211/307; doses 0/0.5/1. No input training or retrospective model tuning.
- Predeclared comparison methods: SCEPT median full-dose signed specificity, traditional unpaired occlusion, simplified low-resolution RISE (256 random binary masks), random permutation, and no-nuisance SCEPT ablation.
- All five rankings assessed on the same zero-fill deletion/insertion true-class softmax probability AUC. This can favor occlusion-like methods and is an out-of-distribution perturbation proxy, not medical truth.

## Genuine numeric observations (n=98 paired original images)

| Method | Deletion AUC ↓ | Insertion AUC ↑ |
|---|---:|---:|
| SCEPT U10 patchwise ranking | 0.336627 | 0.755491 |
| Occlusion | **0.143759** | **0.932104** |
| RISE-like | 0.164482 | 0.923892 |
| Random negative control | 0.578550 | 0.534039 |
| SCEPT no-nuisance | 0.337371 | 0.755842 |

For pairwise normalized improvement, **positive means SCEPT is better**; negative means SCEPT is worse. Paired class-stratified bootstrap 2,000 resamples, paired sign-permutation 2,000 draws and 6-comparison Holm correction:

| Comparator | Outcome | SCEPT minus comparator (improvement convention) | Bootstrap 95% CI | Holm-adjusted p |
|---|---|---:|---|---:|
| Occlusion | Deletion | -0.192867 | [-0.248933, -0.132689] | 0.002999 |
| Occlusion | Insertion | -0.176613 | [-0.219671, -0.133918] | 0.002999 |
| RISE-like | Deletion | -0.172145 | [-0.219127, -0.126112] | 0.002999 |
| RISE-like | Insertion | -0.168401 | [-0.216421, -0.121886] | 0.002999 |
| Random | Deletion | +0.241923 | [+0.186070, +0.300897] | 0.002999 |
| Random | Insertion | +0.221451 | [+0.159417, +0.279117] | 0.002999 |

**There are four significantly unfavorable results vs existing XAI baselines and two favorable results vs random.** Permutation Monte Carlo p values have finite resolution (minimum raw about 0.0005); samples are from one data source.

## Research figures reproduced from the original frozen result tables

These charts are generated directly from the 100-case original numeric records and stored permanently in GitHub. They show **unfavorable SCEPT comparisons**, not a promotional illustration:

- [Actual deletion AUC chart](../results/u11/registered-v0.1/figures/deletion_auc.png)
- [Actual insertion AUC chart](../results/u11/registered-v0.1/figures/insertion_auc.png)
- [Paired SCEPT-minus-baseline effects and 95% bootstrap confidence intervals](../results/u11/registered-v0.1/figures/paired_fidelity_forest.png)
- [Per-original-image graded-sensitivity violation distribution](../results/u11/registered-v0.1/figures/dose_violations_by_image.png)

Editable vector equivalents (SVG) plus SHA-256 of each PNG/SVG are alongside the result tables and registered file-hash manifest. Plot source: [scripts/u11_make_research_figures.py](../scripts/u11_make_research_figures.py).

## Falsification, coverage and ablations

- The required zero-dose B=C controlled null passes on all recorded eligible ROI pairs, as expected by numerical construction.
- There were **755 within-seed patch/dose monotonicity violations**, undermining general positive claims for U10 graded sensitivity. These are correlated ROI observations, NOT 755 independent patients.
- Numeric patch pairing mean coverage 99.84%; all 100 fixed cases had at least 80% matched patches. Numeric intensity matching is NOT anatomical matching, label relevance, causal exchangeability or matched biology.
- SCEPT with no nuisance has almost identical mean deletion/insertion AUC to standard SCEPT. On this dataset nuisance repetitions did not produce a clear fidelity improvement.
- The separate U10 unit study rejects stable-but-zero-effect or missing genuine control evidence. These are engineering falsification checks, not trained-medical-model causal experiments.
- **Independent U10 post-nuisance matching audit: PASS.** Across 100 authentic images, checked all **14,676** (target ROI, control ROI, nuisance seed) combinations and **4,892** eligible ROI contrast records: **zero matching-tolerance violations, zero null-dose violations**, without any threshold change. [Actual per-seed matching audit](../results/u11/registered-v0.1/u10_matching_parity.json). This is *numeric* intensity/std matching, never anatomical or biological causal matching.

## Independent rerun and immutable original outputs

The 100-case experiment executed **158,020 actual CPU ONNX forwards** (approximately 13.65s inside the benchmark computation, excluding runner setup and downloads). A separate second execution from the same pinned model, unchanged original data, and code reproduced the *entire case-level JSON byte-for-byte* and the complete aggregate statistics. This demonstrates deterministic within-environment replay, not generalization to a second dataset.

These permanently versioned original results are in GitHub:

- [Exact derived aggregate summary](../results/u11/registered-v0.1/summary.json)
- [All 100 original sample case-level measurements](../results/u11/registered-v0.1/cases.json)
- [Case table CSV](../results/u11/registered-v0.1/cases.csv)
- [Second genuine ONNX rerun equality evidence](../results/u11/registered-v0.1/replay_integrity.json)
- [Raw result file SHA-256 manifest](../results/u11/registered-v0.1/FILE_HASHES.sha256)

No MNIST raw digit pixels, model binary, or private patient inputs were committed; source locations, checksums and original test-case indices are provided for reproducibility.

## Scientific interpretation and current release gate

**Supported, narrowly:** U10 numeric specificity operates on real frozen task-model predictions; matched ROIs, independent noise seeds, graded interventions and true output contrasts are executable; the experiment is reproducible and passes benchmark-data quality checks; patchwise SCEPT ranks better than random for this small predeclared real-data cohort.

**Falsified in this setting:** SCEPT v1 patchwise specificity ranking does NOT outperform occlusion or the reported RISE-like estimator on the specified deletion/insertion proxy; full controlled dose monotonicity does not hold for many ROI/seed tests; the nuisance-control ablation has little impact.

**NOT established:** SCEPT superiority to Grad-CAM, actual original RISE or Integrated Gradients (none run yet); causal/biological explanation of brain lesions; clinically valid perturbations; calibrated statistical certainty beyond current outcome; robustness across independent model/data families; anatomical spatial alignment; human explanatory ground truth or patient-level medical fairness.

**Unmet M4 requirements:** second independently pretrained model and task/data family, true supported gradient baselines, anatomy-/semantics-aware matching beyond the verified numeric U10 parity audit, model-parameter randomization tests, grounded feature causal controls, independent holdout/generalization evidence and reviewable uncertainty estimates. The grade of M4 is **NOT PASSED** until substantive additional evidence, not just CI, satisfies these gaps.

**Do not** retune U10 v1 thresholds using these results. If improving SCEPT, register new version, baseline strategy and untouched holdout first. Publishing a useful negative research finding is preferable to claiming unmeasured superiority.
