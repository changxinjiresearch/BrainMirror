# SCEPT / SCCA v0.8 官方 MONAI 自动跨层回路搜索

- 交付：官方 MONAI 1.5.1 的真正多 Block 受控串行路径，ROI 驱动的逐跳 Attention Head/Query/Key 自动筛选、实际 Patch Merging 和 Decoder Skip 冻结、输出破坏/恢复测量和末端路径阻断负控。
- 严格区分：**受控计算路径** != **唯一自然因果机制**；当前方法尚未在训练好的 BraTS2020 模型和独立病例上验证。
- 本地运行不依赖 MONAI 的回归及人工图验证；真实官方 MONAI 1.5.1 的不跳过集成测试在 GitHub Actions 上执行。
- 默认多层搜索是有预算的非穷尽筛选，不能宣称找到了所有可行路径。
- 代码：`brainmirror/scept/monai_crosslayer.py`；测试：`tests/test_scept_monai_crosslayer_v08.py`, `tests/test_monai_integration_v07.py`；CI：`.github/workflows/monai_scept_integration.yml`。
