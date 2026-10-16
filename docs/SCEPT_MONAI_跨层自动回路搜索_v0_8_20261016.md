# SCEPT v0.8 — 官方 MONAI SwinUNETR 多层自动受控路径发现

**日期：2026-10-16。状态：工程 Alpha；不构成患者、机制唯一性或优于 Grad-CAM 的科研结论。**

## 1. 开发范围

- 实际运行 MONAI **1.5.1** `SwinUNETR` (明确锁定版本，保证使用官方 `WindowAttention`, `SwinTransformerBlock`, `BasicLayer`, `PatchMerging`, Decoder)。
- 用户只提供原始体积、参考体积、固定的三维 ROI、目标输出类别；算法**自动**确定候选 `window / head / query / key`。
- 捕获所有 Swin Block 基线输入/输出、窗口输入、MLP 输出、真实 shift masks、三维窗口坐标、Decoder skip 第二参数；拒绝运行中形状/掩码不一致、乱序、重复的路径。
- 排列向前的二跳或三跳路径，按有界 beam/精确干预预算测试。先使用主参考选择，再使用第二参考只对最终固定路径复验，不以第二参考重新调参。

## 2. 明确定义的隔离语义

从真实 MONAI 的参考体积 `X_ref` 开始。在第一条指定注意力消息位置，计算**干净输入**的对应消息并替换参考消息。在第二跳及以后，使用上一跳产生的混合张量计算指定原子的消息。每个选定 Attention Block 均将其非选定 Attention 消息和整个 MLP 输出保持在参考值，并在 Block 输出处减掉**进入该 Block 的残差恒等传递差值**，避免该恒等路径绕过指定 Attention 原子。未选中的 Block 只传递其输入残差差值，冻结其非选定 Attention/MLP 变化。层间合法 Patch Merging 运行原版真实计算；解码器 Skip Connection 的第二输入被替换为参考运行中的张量，避免绕过已指定的瓶颈路径。完整解码器照常执行。

核心路径负控：**在最终被选中原子的位置改回参考消息**，此前一路的残差恒等变化应被阻断，输出应数值上恢复参考推理。这个验证不可省略，否则不能称为受控串行路径。若负控失效，一律输出 `unresolved / identity_bypass_failed`。

**重要边界：** 在模块钩子和定义的截断干预下，该算法测量一类人为指定的 **controlled cascade**。MONAI 的全计算图包括卷积、Norm、非线性、重叠感受野及多条旁路，故**不**据此宣称唯一自然因果回路，**不**声称严格保留完整模型的自然数据分布。

## 3. 自动发现算法

1. 固定原始三维 ROI 和输出类别；两次原版 MONAI 前向构建 `clean/reference` 观察快照。
2. 候选 Query 来自准确的窗口/移位映射到三维模型空间后与下采样 ROI 的交集。绝不把 Shift Mask 禁止的 Q→K 链接列为候选。
3. 使用实际差分 Attention 中的 `weight × ΔV` 或 `Δlogit` 进行启发式排序；这只是**候选筛选**，不是归因证明。
4. 起点本身也自动选择：在前 `max_sites` 个可达官方 Block 中筛选具备后继 Block 的候选起点，按 ROI 内干净输入/参考输入的 QKV 差异择优；后续可跳过中间 Block，但必须按实际执行顺序前进。每一跳均从**前一跳受控干预运行**的混合张量寻找候选，并重新执行真实前向完成验证。
5. 所有测试候选路径记录已测的数值效应、逐跳原子张量差异、搜索消耗预算；返回一个原始参考内排序最佳的路径。
6. 执行“末端截断应失效”的负控；失败、预测效应近零或没有合法候选必须弃权。
7. 如果请求第二参考，最终**固定**路径仅做一次保留复验；两种参考方向相反则报告 `conflicting`，绝不重新筛选。

默认筛选资源界限：每个 Site 扫描最多 **96 个 ROI Query 位置**；每 Head/Query、每候选干预类型仅保留最多 **12 个候选 Key**；最多 5 次（烟雾示例）或明确配置的有界真实前向干预。范围并非穷尽式发现。CPU/GPU/显存开销需要跟随分辨率进行预算，不能无界缓存大型患者卷。

## 4. 工程验收门槛

- 源码：`brainmirror/scept/monai_crosslayer.py`。
- 不依赖 MONAI 的解析模型受控路径正负控：`tests/test_scept_monai_crosslayer_v08.py`。
- 真实 MONAI 1.5.1：`tests/test_monai_integration_v07.py` 增加末端路径阻断、跨 Block 及自动筛选测试。
- 可重复烟雾脚本：`scripts/monai_v08_smoke.py`；云端独立运行并发布人工数据审计 JSON，之后复制进 `runs/`。
- 一切数字结论以实际 GitHub Actions 的公开 Job/Artifact 为准。无临床实验、无训练好的 BraTS2020 权重、无 Grad-CAM 对照。

## 5. 仍未解决

- 多种 `downsample` / `use_v2` / checkpointing / nonzero dropout / MONAI 版本的完整兼容性未验证，接口明确拒绝不支持的情况。
- 官方 MONAI 大输入上的性能优化（受控批处理、张量流式记录、拓扑稀疏索引）；真正自动全图搜索跨 Decoder/所有 Skip 的复合路径。
- 训练模型的 MRI ROI 参考质量评估、分布外筛查、稳定性、置信区间、随机种子、真实病例独立验证。
- 与已有 XAI 算法严格对照实验；SCEPT 的科学新增价值并非工程测试的推论。
