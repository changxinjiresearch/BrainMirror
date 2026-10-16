# BrainMirror SCEPT · SCCA v0.6

- 日期：2026-10-16
- 目标：多层 3D Swin 风格真实窗口计算捕获及 ROI→跨层自动机制候选搜索
- 科学状态：工程 Alpha，人工合成数据；没有正式 MRI / 预训练 SwinUNETR / 独立患者验证

## 交付

- `brainmirror/scept/swin3d.py`：W-MSA / SW-MSA、真实 Q/K/V 与 mask/token ID、两尺度各至少两个 Block、Spatial VIB 与 Decoder、执行拓扑导出。
- `brainmirror/scept/swin_discovery.py`：自动源 token + 真实窗口原子 Beam Search、多层精准干预、双向破坏/恢复、第二参考条件复核、预算控制和证据状态。
- `tests/test_scca_swin_v06.py`：独立 SDPA 数值等价、移位伪连接负控、路径无效保护、冻结 skip、已知均匀 Value 消息正控、双精度四层端到端正控、零化第一层 Attention 的机制负控、可复现 JSON。
- `scripts/scca_swin_v06_smoke.py` + `runs/scca_swin_v06_synthetic.json`：可重现合成实验与全部路径记录。
- `docs/SCCA_多层Swin_真实图_自动跨层搜索_v0_6_20261016.md`：方法、技术语义、验收标准和明确未完成项。

本轮未改动旧 BrainMirror 3D Autoencoder、旧 v0.1–v0.5 API 或正式 MRI 数据。

## 实验解释边界

所有选择都是预算限制下的候选；自动搜索不等于 exhaustive 机制发现。Attention 的 causal atom 是人为隔离后的可检验计算效应，不是模型唯一真实机制；随机初始化合成模型若没有可靠复验，应输出未解决状态。独立真实 SwinUNETR/MONAI 适配和患者级科研验证均留待后续。

## 本地最终验收

- `python -m pytest -q tests`: 88 passed（2026-10-16 本地 CPU）。
- 自动发现烟雾实验：136,224 次原子候选资格/近似筛选计数，3 条完整路径双向干预，第二参考未形成支持，结果为 `unresolved_or_conflicting_references`；不声称已找到可靠医学解释。
