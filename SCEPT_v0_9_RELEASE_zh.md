# SCEPT v0.9 — BraTS2020 真实医学实验管线

- 代码阶段：正式 MONAI 1.5.1 的 BraTS2020 四模态 NIfTI 导入、受试者级划分、训练、滑窗分割评估、模型权重审计、局部 ROI 参考干预、SCCA 搜索。
- 公开测试阶段：CPU 单测与官方 MONAI 合成 NIfTI 集成测试（只验证程序行为，不构成 MRI 学术证据）。
- **BraTS2020 患者级科学验证：没有真实病例数据或合格训练权重，因此尚未执行，不能声称已通过。**
- 代码 `brainmirror/scept/brats2020.py`, `brats_v09_pipeline.py`; CLI `scripts/scept_brats_v09.py`。
- 工程可复现材料：`tests/test_brats_v09.py`, `tests/test_brats_v09_monai.py`, `scripts/scept_brats_v09_synthetic_smoke.py`, `docs/SCEPT_BraTS2020_v0_9_实验协议与状态_20261016.md`。
- 数据：请使用先前具有文件审计的 BraTS2020 Kaggle 执行环境输入路径；不要将 MRI 或权重提交到 public GitHub 仓库。
- 严格拒绝将 Synhetic smoke 的 Dice、SCCA Unresolved 或参考一致性宣称为真实训练模型的患者级机制证据。
