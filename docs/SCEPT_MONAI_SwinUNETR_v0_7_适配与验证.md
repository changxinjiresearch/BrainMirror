# SCEPT v0.7 — 官方 MONAI SwinUNETR 适配 / 工程规格

日期：2026-10-16。软件实现阶段：**MONAI 1.5.1 特定版本适配，非临床验证**。

## 1. 版本范围与依赖

- 依赖官方 `monai==1.5.1`、`einops`、PyTorch 2.x，实际官方库的 CI 由 `.github/workflows/monai_scept_integration.yml` 执行。
- 不替换或复制 MONAI 训练权重，也不在源码中修改 `SwinUNETR.forward`、`WindowAttention.forward`。
- 未经验证的 MONAI 版本会拒绝构建；兼容其他版本必须新增官方库回归测试。
- 当前本地研发环境没有 MONAI，**本地数值单元测试使用独立、结构相似的假模块**。GitHub 官方库集成测试结果需单独审计。

## 2. 新增模块

`brainmirror/scept/monai_adapter.py`

1. `require_monai`：懒加载并检查版本。
2. `MonaiSwinObserver`：枚举实际 WindowAttention；记录真实窗口输入、shift mask、张量形状和父 Block 空间布局；Hook 在异常或正常退出后均清理。
3. `decompose_monai_attention`：对 MONAI 的真实 QKV、scale、relative-position bias、shift mask、Softmax、Attention×V、输出投影做只读重算。
4. `AttentionAtom` 与 `patched_attention_output`：一次替换指定窗口/Head/Query/Key 的加权 V 消息，或指定 QK logit 并重新归一化整行 Softmax。对于被 shift mask 禁止的连接，拒绝干预。
5. `WindowGeometry`：根据 MONAI 实际 Block 维度、窗口大小、移位、填充，返回每个窗口 local token 的原始 3D 位置。padding token 返回 `None`。
6. `atomic_bidirectional_effect`：调用完整官方模型前向，记录指定原子替换造成的破坏/恢复效应；Decoder 和 Skip 仍由真实模型执行，但**其他未选层不冻结**，因此不宣称完整跨层路径隔离。
7. `MonaiDecoderTrace` 与 `substitute_decoder_skip`：捕获多尺度 encoder/decoder 实际张量及单个解码器 Skip 输入替换。Skip 输入干预属于节点/输入干预，非唯一通路证明。

`brainmirror/scept/monai_vib.py`

1. `MonaiSwinVIBBridge`：复用官方 `SwinUNETR` 的所有编码器、Skip 与 Decoder，仅在最深层 `hidden_states_out[4]` 前加入 **空间高斯 VIB**。VIB 关闭时必须与官方 `SwinUNETR.forward()` 数值等价。
2. 可用确定性 `z=mu` 或传入固定噪声 `epsilon` 采样；保存 `mu / logvar / KL` 供干预记录。
3. **重要约束：** 多尺度 Decoder skip bypass VIB，不能把 VIB 当作全部信息的唯一瓶颈。

## 3. 主要数值验收

- 官方 MONAI `WindowAttention.forward` vs 独立公式复算，带/不带 shifted-window mask 的最大绝对误差目标 ≤ 2e-5。
- 关闭 VIB 的桥接 forward vs 原模型 forward 的最大绝对误差目标 ≤ 1e-6。
- 原子替换：同一参考状态导致零变化；不同参考可记录带符号效应；Hook 结束后原模型输出不变。
- 未获准的 Q/K pair 会抛错，不能把被屏蔽连接当作真实可传播路径。
- 解码器 Skip 输入替换可测得预测变化，干预 Hook 自动恢复。
- 使用 32³ 人工输入 / batch normalization 的小型 SwinUNETR 只是 **CI 工程验收**。不代表 BraTS2020 临床分割、I-JEPA 预训练或完整论文有效性。

## 4. 真实数据与科学结论边界

尚未：导入 BraTS2020 患者 MRI；加载真实训练模型权重；进行医学合理参考替换；开展患者级复核；全图跨层边冻结；报告 Grad-CAM 对照与可信度统计。

**v0.7 的目的仅是把 SCEPT 的基础监测、原子干预和 Spatial VIB 桥接接入真实第三方 MONAI 实现。** 完整 SCCA 自动搜索当前仍运行在此前的自研轻量 Swin 模型上，不应被描述为已经自动发现官方 MONAI SwinUNETR 的全路径回路。

## 5. 操作与 CI

```bash
pip install 'torch>=2.3,<3' 'monai==1.5.1' 'einops>=0.7,<1' pytest numpy
pytest -q tests/test_monai_integration_v07.py
python scripts/monai_v07_smoke.py --json-out runs/scept_monai_v07_synthetic.json
```

来源：MONAI 1.5.1 `monai/networks/nets/swin_unetr.py`（Apache-2.0）；MONAI 官方网络说明。所有代码为自行实现的适配/观察层，未拷贝 MONAI 源码。

## 6. ROI 驱动的 MONAI 原子候选筛选（补充）

`rank_monai_roi_atoms` 可将三维输入 ROI 按实际 stage 大小映射为候选 Query token，利用真实窗口空间索引与 shift mask 过滤不合法连接；按加权 Value 消息的绝对范数进行排序。此排序只是成本较低的**候选筛选启发式**，不能被当作因果解释。`verify_ranked_monai_atoms` 对限定数量的候选执行真实的完整模型正反向原子干预，并逐个记录效应。这仍不是 MONAI 版本的全局跨层回路联合优化。
