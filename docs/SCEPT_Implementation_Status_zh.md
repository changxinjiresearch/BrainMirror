# SCEPT Alpha + Transformer-VIB 集成：工程状态

## 2026-10-16 本次实现

新增 `brainmirror.scept`，与现有 BrainMirror 3D 卷积 AE 隔离，不改变既有 API。

- `TransformerVIB3D`：四模态 3D patch tokenizer、真实 MultiheadAttention 的 Transformer、空间 VIB、3 路重叠分割 logits；合成输入可训练。
- `IJEPAPretrainer`：三维遮挡 token 预测 / EMA Teacher 的 **I-JEPA-style** 自监督示范；仅迁移学生编码器，**不是**论文 I-JEPA 完整复现，不是正式 MRI 预训练权重。
- `trace`：真实模块输出与 Attention 矩阵捕获、等价性测试、Attention/MLP/VIB 的节点级状态替换。
- `explain`：固定输出 ROI、合成均值参考空间干预、内部模块状态破坏/恢复效应、机器可读 JSON。
- `scripts/scept_smoke.py`：在人工合成数据上演示预训练、权重迁移和双模型 SCEPT 解释。

### 科学边界

**完成工程 Alpha 和轻量 Transformer-VIB 集成；未科学验证 SCEPT、未证明优于 Grad-CAM，未使用真实脑 MRI。**

1. 捕获的 Attention 权重不是因果解释；运算图也不是三维点线云。
2. 当前仅支持 **node/module output patching**；尚无真实边级/路径级隔离、稀疏 SCCA 联合优化或独立验证，不能称为已证实的 evidence circuit。
3. 当前参考是每模态 volume mean fill，仅用于合成测试，**不可**用于声称医学合理的反事实。
4. 当前全局注意力 O(N²) 且 `max_tokens=512`，BraTS 全分辨率需裁剪或改成 3D windowed Swin。
5. VIB 在确定性 μ 模式下测试干预，训练可随机采样；随机复核与跨参考一致性后续实现。
6. 三通道 WT/TC/ET 独立 logits 尚未强制嵌套一致性，也无临床验证。
7. 本次双模型均**未在真实 MRI 上充分训练**，所有演示为工程测试。

### 本地运行

    python -m pip install -r requirements.txt pytest
    python -m pytest -q tests/test_scept.py
    python scripts/scept_smoke.py --output runs/scept_alpha_smoke.json

### 下一阶段

- 加入 Swin 3D Window Attention / decoder skip tracing，做真正的 BraTS 模型集成。
- 明确区分 node 与 edge/path patching，补全随机对照、参考有效性测试与独立验证。
- 完成 SCCA 联合目标、可复现优化器和负对照。
- 按患者划分真实 BraTS2020，完成 3D I-JEPA 正式预训练与双模型对照、Grad-CAM 公平比较。
