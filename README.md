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

## P1 — IXI 健康脑 MRI 数据入库门禁（进行中）

[P1 数据源与许可](docs/P1_01_数据源许可与选择.md) · [P1 操作 SOP](docs/P1_02_实施流程与验收.md) · [P1 工程验证回执](docs/P1_03_工程验证回执.md)

正式数据源候选 IXI T1（官网 [IXI dataset](https://brain-development.org/ixi-dataset/)，标注 CC BY-SA 3.0）。此仓库是 **Public**，绝对禁止将原始 MRI、可识别病例 ID、私有逐人清单和密钥上传 GitHub。使用前确认许可证、来源和研究用途。

```bash
# 先把下载的 IXI T1 NIfTI 存放在仓库外的私有目录；准备自己的 >=32 字节密钥
export BRAINMIRROR_P1_HMAC_KEY='<your-private-secret-at-least-32-bytes>'
python scripts/p1_intake.py --data-dir /private/ixi-t1 \
  --output-dir /private/brainmirror-p1-audit \
  --accept-ixi-terms --voxel-qc
```

上面只做**真实扫描资格审查**，不做训练。当前未获取 IXI 真实影像，P1 仍处于 `IN_PROGRESS_BLOCKED_ON_REAL_DATA`。首批 P1 代码已有 synthetic fixture 测试，但不代表通过正式科学门禁。

### 旧前端离线依赖（仅本地 P0 兼容）

为了避免把 4.8 MB 的第三方压缩 JS 源重复提交到公开 Git，`web/plotly.min.js` 不跟踪。可从已保存的 P0 初始归档中离线恢复并核验 SHA256：

```bash
python scripts/restore_vendor_plotly.py --p0-zip /path/to/BrainMirror-P0-initial.zip
```

禁止将此旧版带 MRI 上传的 API 直接公开到互联网；新的 P3/P4 前端将使用经安全审查的独立依赖构建。
