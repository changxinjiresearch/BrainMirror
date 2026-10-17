# SCEPT U6 — Automatic Experiment Task Scheduler

**Scope:** browser-local, serial, on-demand scheduling of genuine frozen ONNX inference using the existing U1 model contract, U3 typed feeds, U4 numeric observations, and U5 intervention plugins. U6 is an **experiment orchestrator**, not a trained model, causal inference method, clinical diagnosis, 3D anatomical registration, or scientific interpretation system.

## Code and entrypoints

- `scept-web/core/scept_scheduler.js`: model-agnostic `SCEPT_SCHEDULER.make({session,interventionEngine,observationAdapter,dataAdapter,core,onProgress})`. Task queue with typed statuses, per-task deterministic seed, A/B runs, A/B/C matched-control runs, intensity sweeps, batch plans, cancellation, retries, stable record export and input SHA-256 hashes. No fixed 197×192 tensor assumptions.
- `scept-web/universal.html` + `universal.js`: load a genuine ONNX model, select U5 intervention, provide baseline configuration, seed and explicit intensity list or control variants, build and run queue, inspect individual rows and export JSON/CSV.
- `scept-web/tests/u6_scheduler.cjs`: strict deterministic schedule, stop/resume, partial-run, transient failures, error isolation, retry without changed seed, input hashes and privacy.
- `scept-web/tests/u6_browser.cjs`: **real ONNX Runtime Web** CNN convolution model with a 3-level noise sweep, repeated seeds producing identical hidden tensor observations, matched A/B/C variants, intentional invalid control and retry. The CNN is a deterministic untrained engineering fixture.

## API usage

```js
const queue = SCEPT_SCHEDULER.make({
 session: openedModel, interventionEngine, observationAdapter, dataAdapter,
 core: SCEPT_CORE, onProgress: snapshot => render(snapshot)
});
await queue.create({
 mode:'sweep', pluginId:'u5.noise', parameter:'sigma',
 config:{region:{bounds:[[0,1],[0,1],[0,1],[0,2]]}},
 baseSeed:240216, intensities:[0,0.1,0.2,0.3]
},{feeds:typedModelInputs,profileId:'cnn-nchw'});
await queue.run();
const audit = queue.exportAudit();
const csv = queue.exportCSV();
```

The three scheduling modes:
- `ab`: exactly one real baseline/edited task.
- `sweep`: explicit numeric intensities over `sigma`, `fill` or `value`, each task receiving a deterministic independent seed.
- `batch`: explicit `controls:[{id,config,pluginId?},...]` variants; a `u5.matched-control` variant runs genuine A/B/C under only its narrow numeric matching definition.

### Task model and guarantees

Every task is `pending`, `running`, `completed`, `failed` or `cancelled`. The queue runs serially and catches failures **per task**. `retryFailed(ids?)` requeues only failed jobs and increments attempt count; same task ID, numeric intervention parameters, and deterministic seed are preserved. `resume()` requeues cancelled jobs. `cancel()` is a cooperative cancellation request and cannot interrupt a native ONNX Runtime call already executing; a partially executed job is never falsely labelled completed. `run({maxTasks})` allows bounded manual batches. Up to 128 tasks per plan.

The seed is derived deterministically from a fixed uint32 base seed and original plan index, not from execution ordering, thus unaffected by retries or partial batches. Note: real numerical results are reproducible under the **same model/runtime/environment** within floating-point tolerances; exact bitwise portability across different hardware/ONNX providers is not promised. Task metadata and results do not include model weights or original input arrays. SHA-256 per named typed input and reference tensor is recorded for audit, but **JSON export alone is not a restorable compute checkpoint**.

After a page refresh or browser closure, a running in-memory queue **does not continue in the background**, and its raw tensors are not persisted. To replay, explicitly reload the same original ONNX model and inputs, then recreate the plan. Local-only metadata and deliberate export are preferred over silently storing patient MRI. The user must keep exported audit artifacts for reproducibility.

Errors retain a code and bounded message. No fake zeros or fabricated activations are injected for a failed task. Only real U4 exported ONNX tensors appear in results, including actual A/B/C observations. Reference replacement and internal masks inherit their strict U5 compatibility checks. U6 does not automatically select scientific matching covariates, transform MRI regions, select explanation targets or verify medical validity.

## Acceptance

Gate U6.1 — independent plan parsing, 128-task budget, seed reproducibility, task state transitions, cooperative cancellation, isolated failure and retry.
Gate U6.2 — real ORT/WASM three-step intensity sweep, nonzero perturbation response, exact repeated-seed replay under same browser runtime, input hashes and audit privacy.
Gate U6.3 — genuine A/B/C matching-control ONNX forward passes, a controlled failed task and requeue preserving seed and other successes.
Gate U6.4 — existing U0–U5 regression suite unchanged and still green.

**Maturity:** only mark complete after the GitHub Actions run for the final commit confirms all required tests passed. Deployment to a public site is a separate action.
