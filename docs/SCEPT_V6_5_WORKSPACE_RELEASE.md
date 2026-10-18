# SCEPT V6.5 — Universal Research Workspace (Engineering Release)

## Release scope

V6.5 reorganises the public landing page as **Choose actual model/data → Capture one true forward and NAF → Inspect/write/export run journal → Launch validated, explicitly labelled research tools.** V6.5 DOES NOT automatically implement V6.4; the older independent V5 NDF/CRC/ICM and U10/CEDS workflows still have their own run engines, and the universal ONNX import console currently has a legacy controlled A/B workflow rather than automatic V6 \`captureOnce\` support. These distinctions are stated visibly.

**Privacy and costs:** GitHub Pages and CPU/WASM in the local browser, no additional paid hosted API, no remote patient data upload, and no cloud-stored long-term history. The journal holds at most 32 summary-only records in the current browser page memory. **Export JSON to preserve your research**; closing the page will lose unexported journal entries. Raw patient voxels and raw hidden-tensor arrays are never included in journal exports. This is not an end-to-end clinical dataset policy.

## User-visible page

The default public home routes to \`v6.html\`. An upfront researcher-focused panel offers:
- Real pretrained, pinned source/model-hash ViT Tiny NAF on a public default MNI atlas slice (not MRI task pretraining).
- Real pretrained pinned EfficientNetB3 MRI research model with **actual instrumented Conv outputs**, caveated because the model author reports train/test leakage.
- An honestly labelled **Advanced generic ONNX console** for other architectures, without pretending its legacy A/B path already uses V6 run records.
- A secondary MRI/NIfTI native input panel; no prerequisite perturbation, tumor masking or second MRI for NAF.
- Clear shortcuts to independent NAF, project journal and research protocol catalog.

The 6 protocol routes each open a real internal web path: Observe NAF, MRI input adapter, generic ONNX console, V5 NDF/CRC, v4 U10/CEDS and existing model lab. They are navigation **not** a new comparative experiment engine, scheduler, protocol uploader or independent-science validation. V6.4 remains pending.

## Versioned workspace journal

\`scept.workspace.journal.v1\` collects scientifically labeled **\`scept.run-record.v1\` + \`scept.naf.v1\`**, preserving per-run model SHA, preprocessing SHA, named input hashes and true output SHA. The real \`SCEPT_V6.onRunCapture\` hook passes the **actual completed run record** after exactly one real model forward. This hook does not change the frozen model's numerical outputs or the V6.3 NAF scientific definitions.

Interactive functions:
- List at most 32 independent observations (scientific summaries).
- Select a run to inspect provenance and measured-unit counts.
- Edit a bounded 500-character free-text research note.
- Change project title (only in memory).
- Export a single record or entire bounded project manifest as summary-only JSON.
- Import a saved manifest **atomically** with schema, SHA-256 syntax, run ID, model/input agreement, authentic measured-probe references and finite RMS sanity checks. Imported entries are ALWAYS \`imported_unverified\` even if source claims otherwise. No import can recover or claim a full original patient MRI or full tensor arrays. Journal content is untrusted unless re-executed with verified original model/data and matched preprocessing.
- Remove an entry or clear the current in-memory session after user confirmation.

This is versioned data organization and **not** encrypted storage, remote persistence, biometric anonymization, cryptographic proof of a third-party inference, attribution, statistically independent replication, or a verified explanation.

## Non-negotiable science rules

NAF magnitude and shape are actual internal **numeric state** only, not saliency, importance or causality. V6.3 computation updates remain gated by real same-forward, same-coordinate input/output probes. References require genuine distinct input SHA and a fixed frozen/preprocessing identity; examples from slices of one MRI are descriptive **within-volume** ranks, never independent patient reference. No automatic significance, brain tumor diagnosis, trustworthy medical conclusion or SCEPT Research M4 achievement. The U11 negative scientific result remains.

## QA and public deployment

\`node scept-web/tests/v65_workspace_contract.cjs\`: summary-only journal, atomic unverified JSON import, bounded capacity and malicious/raw data rejection.

\`node scept-web/tests/v65_workspace_browser.cjs\`: real public pretrained ViT, actual public NIfTI, one true NAF run automatically reflected in the journal; test notes/export/download/import source downgrade, genuine model routing and six functional protocol links. Captured screenshot is a *real executed browser* visual artifact, not a concept image.

\`node scept-web/tests/v6_23_naf_browser.cjs\`: regressions for real ViT NAF, measured 197-Token residual update, six distinct MRI slices, five-item empirical reference and real instrumented CNN Conv hidden probes.

Original V5, U10, Universal Core, U4 and scientific CI must remain green. Home-page switch occurs **only after all tests pass** via existing zero-additional-fee GitHub Pages workflow. v5.html, v4.html and universal.html retain their separate URLs.

## Deferred V6.4 & V6.6

V6.4: independent Run Record identity/alignment and shared scheduler/protocol SDK for NDF, CRC, ICM and U10; currently NOT COMPLETE. V6.6: cross-model science/performance/reproducibility and rigorous failure reporting; NOT COMPLETE. Workbench V6.5 shipping does not fulfill them implicitly.
