# SCEPT/SCCA v0.7 — 官方 MONAI SwinUNETR 计算与 VIB 适配

- 官方 MONAI v1.5.1 版本门控。
- 真实 WindowAttention (QKV/relative bias/shift mask/projection) 数值重算。
- Head–Query–Key 定点原子干预：weighted Value message 或 QK-logit/Softmax 重归一化。
- 实际三维窗口索引反演（含 shift、padding）。
- 多尺度 Encoder/Decoder/Skip 张量追踪及 Skip 输入干预。
- Spatial VIB bottleneck 透明桥接：关闭时 forward 必须与官方模型等价。
- 官方库 CI 专项验收：合成 32³ 数据，不依赖 BraTS2020。
- 限制：尚未实现 MONAI 整体自动跨层路径搜索；不可当作经过医学解释验证的版本。

详见 `docs/SCEPT_MONAI_SwinUNETR_v0_7_适配与验证.md`。
