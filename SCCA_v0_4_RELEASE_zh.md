# SCEPT / SCCA v0.4：Attention 原子计算与多尺度路径

- 集成对象：`TransformerVIB3D` + 全新 `MultiScaleTransformerVIB3D`（CPU 可测原型）。
- 原子单位：指定 Head、Query token、Key/Value token 的 post-softmax 加权 Value 消息；或 pre-softmax 单 QK score（明确标注 Softmax 行联动）。
- 原子跨层路径：显式 residual source → Attention message → downstream MLP → Spatial VIB → 真实 Decoder。中间非目标分支冻结为参考值。
- 空间潜在干预：指定 Spatial VIB 的 `z[token,channel]`，以独立 ConvTranspose3d 线性算子等价性证明下游变化。
- 多尺度：4→2 网格 Patch Merge，真实 coarse Attention & VIB、fine Skip / upsample / Decoder；支持指定一个 fine token 至单个 coarse attention message 的受控路径。
- 候选自动化：按输入指定的 Head/Query/Key 网格有界枚举，报告全部效应而非只显示最好的一条。
- 测试：完整本地 pytest + 固定 seed 合成 smoke + GitHub Actions；请以实际日志为准。
- 非目标：未训练 BraTS2020，未完成 Swin/W-MSA/shifted attention，未完成真实 MRI 空间干预可靠性，未证明优于 Grad-CAM，未获得唯一的自然因果路径。

## 查看

- 原理与可重复性：`docs/SCCA_原子注意力_空间VIB_多尺度_v0_4_20261016.md`
- 工程 smoke：`scripts/scca_atomic_v04_smoke.py`
- 存档：`runs/scca_atomic_v04_synthetic_smoke.json`
