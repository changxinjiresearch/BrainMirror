# BrainMirror · SCEPT / SCCA v0.5 — 自动多尺度机制发现

- 版本：v0.5 engineering-alpha，2026-10-16
- 目标：用户只指定 MRI 3D 区域掩膜，不再手动填写 Head、Query、Key、fine token 或路径。
- 具体路线：MRI 区域干预 → fine-scale 全局表征中介筛选 → 2× Patch Merging → coarse Attention Head–Q–K 原子消息筛选 → 精确双向干预 → Spatial VIB + Decoder → 保留第二参考策略的独立候选复核。
- 路径对象：适用于 **2-scale 3D full-attention prototype，coarse depth=1** 的显式计算干预图。**不支持**直接推断 Swin shifted-window、多 coarse block、或唯一自然因果路径。
- 输入参考：模糊与 ROI 外通道均值，两者均仅作为**合成诊断参考**，不视为医学可信反事实。
- 正控：固定种子 2025 的 3D 合成 smoke，实际 1,024 个原子候选打分、20 个精确干预、5 个保留参考复核，至少一条通过同病例方向一致性阈值。此结果不代表医学模型或独立患者效度。
- 数值验证：与 v0.4 独立原子干预 oracle 对齐，梯度筛选排除 fine-skip 分支，随机恒定图像必须 abstain，禁用 coarse attention 输出必须 abstain，所需计算预算严格受限。
- 本地测试：`python -m pytest -q tests` → 70 passed（CPU synthetic engineering tests）。

## 文件

- `brainmirror/scept/auto_discovery.py`
- `tests/test_scca_auto_v05.py`
- `scripts/scca_auto_v05_smoke.py`
- `runs/scca_auto_v05_synthetic_smoke.json`
- `docs/SCCA_自动化多尺度机制发现_v0_5_20261016.md`

## 快速示例

```python
import torch
from brainmirror.scept import AutoDiscoveryConfig, discover_scca_multiscale
from brainmirror.scept.multiscale import MultiScaleTransformerVIB3D

model = MultiScaleTransformerVIB3D(dim=8, heads=1, patch=2, latent_dim=8).eval()
x = torch.randn(1, 4, 8, 8, 8)  # 合成数据：并非 MRI 病例
roi = torch.zeros(8, 8, 8, dtype=torch.bool)
roi[:4, :4, :4] = True
report = discover_scca_multiscale(model, x, roi,
    config=AutoDiscoveryConfig(max_exact_tests=12))
print(report['status'], report['best'])
```

## 重要边界

已实现“自动发现候选且按真实计算干预验证”的**可运行方法原型**，并非已证明的医学 XAI 机制。待下一轮完成多 coarse blocks、W-MSA/SW-MSA 的合法索引与链式干预，之后进行真实受试者独立验证及 Grad-CAM / Occlusion / path-patching 对照。不能把 `same_case_cross_reference_supported` 翻译成“因果已被证明”或“临床解释有效”。
