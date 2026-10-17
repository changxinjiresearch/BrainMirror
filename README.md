> **SCEPT U8 universal data-driven visual research workstation:** [engineering design/acceptance](docs/SCEPT_U8_UNIFIED_VISUAL_WORKSTATION.md). Uses real U5/U6/U7 experiment results to drive shared-scale 3D ONNX graph/Token/Channel A/B/Δ displays and explicit native-axis typed input (including rank-5 volumetric) slices, without inventing activations or assuming medical anatomy. Legacy v3.1 is preserved. Check final CI before asserting acceptance.

> **SCEPT U7 / M2:** [XAI plugin and evidence protocol](docs/SCEPT_U7_M2_XAI_PLUGIN_EVIDENCE.md). New declarative experiment JSON specifies pinned model, actual target/observations, U5/U6 intervention plan, independently registered explanation algorithms, and per-task / suite numeric evaluations. A synthetic sparse-mask lesion-removal hypothesis is supported; **medical/causal results are not established**. Check final-head CI before declaring M2 verified.

> **SCEPT U6 on-demand experiment scheduler:** [engineering contract](docs/SCEPT_U6_EXPERIMENT_SCHEDULER.md). Real A/B, explicit intensity sweeps, numeric matched A/B/C batches, deterministic task seeds, status/retry/cancellation and JSON/CSV audit. Code completion is distinct from green CI and public website deployment.

> **SCEPT U5 universal intervention engine:** [engineering contract](docs/SCEPT_U5_INTERVENTION_ENGINE.md). Includes seeded Gaussian noise, tensor ROI occlusion, explicit reference replacement, numeric matched control A/B/C and genuine in-graph internal masks where a compatible ONNX graph is instrumented. U5 scientific validity remains unverified; check CI before asserting U5 gate passed.

> **SCEPT U3–U4 M1 architecture-independence development** is in branch `phase/scept-u3-u4-m1-architecture-independence` with [U3/U4 acceptance contract](docs/SCEPT_U3_U4_M1_ARCHITECTURE_INDEPENDENCE.md). Do not call M1 verified before CI browser evidence. The older ViT 197×192 visualisation remains isolated in its model-specific legacy path.

> **SCEPT Observatory v2 browser-first preview** is under [scept-web/](scept-web/) with [engineering acceptance contract](docs/SCEPT_BROWSER_PLATFORM_V2.md). It is **training-independent** and does not need another model training run; existing P0–P8 training-oriented docs below are retained as historical BrainMirror plans, not SCEPT prerequisites. Full science/production validation is NOT complete. I-JEPA is explicitly excluded.\n\n## SCEPT Browser Studio (separate from model training)

**Online research studio:** https://scept-studio-brainmirror.vercel.app  
**Source:** [scept-studio/](scept-studio/)  
**Audit:** [SCEPT v1.1 engineering gates](docs/SCEPT_BROWSER_STUDIO_V1.md)

SCEPT now has its own browser-first, frozen-pretrained-model interface. It is designed for ONNX model import, real CPU/WebAssembly inference, visible exported internal tensors, paired numeric interventions, audit traces and local data handling. This **does not mean universal model compatibility or scientific validity is complete**. I-JEPA is deferred. The original BrainMirror P0 training-first documents below refer to the separate legacy MRI learning research.

---

# BrainMirror — AI Black Box Observatory

**交互式神经网络机制观测研究平台**：训练模型学习真实正常脑 MRI，左侧修改三维脑结构，右侧直接查看**真实计算单元、真实连接、激活和干预效应**。目的是形成可检验的 AI 内部计算机制假设，而非播放动画或模拟生物神经电活动。

> 当前状态：**P0 本地工程基础**。历史 v0.1 模型仅在合成体素 phantom 上训练，32³ 分辨率，不代表正常人脑解剖学习；v0.2 浏览器为真实 PyTorch 的**预计算记录**，非网页实时推理。真正的健康 MRI 训练、在线实时科研 API 和机制验证仍未完成。

## 阅读入口

1. [完整功能规格（73 项）](docs/01_完整功能规格.md)
2. [P0–P8 详细开发计划](docs/02_详细开发实施计划.md)
3. [科学实验协议（H1–H3）](docs/03_科学实验协议.md)
4. [架构、API、节点/边数据字典](docs/04_架构_API_数据字典.md)
5. [GitHub 初始化与协作流程](docs/05_GitHub工作流与立即操作.md)
6. [隐私与数据治理](docs/06_数据治理与隐私.md)
7. [当前状态与历史限制](docs/07_研究与工程状态.md)

## 本地科学计算内核（不需要云端）

Python 3.11+，需要 PyTorch / FastAPI / SciPy。项目保留旧版完整 web 资源用于回归参考，但未经鉴权不应暴露为公网服务。

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
python scripts/p0_verify.py --output runs/p0_smoke.json
uvicorn brainmirror.server:app --host 127.0.0.1 --port 8000
```

浏览器本地地址：`http://127.0.0.1:8000`。请**不要**使用 `--host 0.0.0.0` 将目前可以上传 MRI 的服务公开到网络。

`p0_verify.py` 使用**完全人工合成 phantom** 检查实际 PyTorch 推理、加权连接数值一致性、局部变形和内部门控干预。它的 PASS 只代表工程数值 QA，不代表科学发现。

## 真正科学研究的硬条件

- 真实、许可合规的健康 T1 MRI；受试者级数据划分与防泄漏。
- 根据正式分辨率与算力重新训练模型；32³ 工程基线不可被视作微细脑区解剖可靠性的证据。
- 变形必须有空间和解剖 QC；显式区分 geometry-only 和 intensity-only。
- 不能假定神经元对应人类脑区；每个模型计算单元以实际 layer/channel/spatial index 索引。
- 禁止把所谓“相关节点连线”误标为真实卷积加权连接，或把单节点消融效应解释为完整因果机制。
- 公共仓库不得提交 MRI 原始数据、受试者身份、Token 或医学隐私。

## 项目结构

```
BrainMirror/
├─ brainmirror/          # 本地 PyTorch & API 核心
├─ web/                  # 旧版 local-first UI，待 React/NiiVue 重构
├─ scripts/              # P0 smoke / P1 headers inventory
├─ tests/                # 数值、API、工程测试
├─ docs/                 # PRD、协议、计划、治理、仓库流程
├─ .github/workflows/    # 工程 CI
└─ README.md
```

## 授权与发布

当前尚未最终选择源代码开源许可证，也没有授权公开再分发任何 MRI。仓库建议初始设置 Private；之后单独决定版权许可、伦理和发行策略。旧版 Plotly 前端 JS 附有相应第三方许可证。数据源许可请逐个独立核对。
