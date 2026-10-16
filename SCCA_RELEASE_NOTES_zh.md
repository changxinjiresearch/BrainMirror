# BrainMirror SCEPT / SCCA 工程版 v0.2

- 日期：2026-10-16
- GitHub 分支：`phase/scept-transformer-vib-alpha-20261016`
- 状态：合成数据工程正确性验收（19/19 本地测试通过）；**不是**科研/临床有效性证据。

## 本次完成

1. 将真实 Transformer 自注意力 / MLP 输出沿独立残差分支的消费路径显式暴露为两个 `Identity` EdgeGate；原有模型参数和正常推理保持不变。
2. 增强 `trace.instrument`：同时干预多条已声明边、指定 token 掩膜、严格检查形状/类型/设备、自动清理 Hook。
3. 新增 `brainmirror/scept/scca.py`：自适应候选梯度近似筛选、真实两向干预、稀疏 + 非目标 + 破坏/恢复联合目标、受限 Beam Search 和保留参考复验。
4. 新增已知依赖的合成基准：零贡献 decoy、无效应负控、双边协作、效应正负号、参数不变性、捕获等价性与不同参考策略等。
5. 提供 `scripts/scca_smoke.py` 及可追溯 JSON 结果。

## 无法宣称的内容

- **尚未**实现 arbitrary attention-head/token-to-token/multi-hop 的精确路径干预；当前结果是声称范围有限的 residual edge-set interventions。
- `channel_mean` / `local_blur` 是合成测试参考，不等于医学合理的 MRI counterfactual。
- 保留参考复验使用同一个合成数据体素块，不是独立患者外部验证。
- 尚无临床数据、BraTS2020 实验、Grad-CAM 对照、外部验证或原创性证明。

## 快速运行

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests/test_scept.py tests/test_scca.py
python -m scripts.scca_smoke --output runs/scca_synthetic_smoke.json
```

需在已有 BrainMirror 仓库根目录执行；压缩包是增量代码（overlay），不会包含原始患者数据。
