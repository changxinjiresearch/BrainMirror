# BrainMirror public IXI anatomical reference

This directory is for **one sanitized group-only browser preview**, not individual MRI scans.

Expected file: `ixi_population_467.json` (the SHA256-pinned, 467-subject mean MRI visualization candidate).

**Source:** IXI Dataset, Brain Development — https://brain-development.org/ixi-dataset/

**Data license:** Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0), https://creativecommons.org/licenses/by-sa/3.0/.

**Derived data / changes:** 467 IXI training-set T1 scans underwent SynthStrip brain extraction, 2 mm affine spatial registration to one reference training subject, intensity normalization, masked group averaging and quantized browser-preview downsampling. The resulting group template is an **unreviewed scientific candidate**, not an unbiased nonlinear atlas, not anatomical-region segmentation, and not functional whole-brain simulation.

**Privacy:** Never upload raw or individual MRI images, private split manifests, subject IDs, demographic attributes, original per-subject QA receipts or Kaggle tokens to this public directory. The only allowed data asset is an anonymized, allowlisted group preview.

**Provenance / integrity:** The public viewer pins the exact sanitized export SHA256:

`36b25fbf1f6749b309cc93adc811ee2137fa83fdab89cf72ba87886e542cda21`

Owner-controlled source-of-truth for the public file:
`BrainMirror-Data-Private/staging/ixi-public-preview/ixi_population_467.json`

Licensed derived data remain CC BY-SA 3.0; application source code retains its separate licensing terms.
