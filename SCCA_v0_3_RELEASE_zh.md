# SCCA v0.3 — 精确声明路径（Engineering Alpha）

**日期：** 2026-10-16  
**状态：** 33/33 本地单元测试通过。所有数值来自未训练小型 Transformer-VIB 上的合成随机数据，非真实 MRI 研究结果。  
**位置：** BrainMirror 功能分支 `phase/scept-transformer-vib-alpha-20261016`。

## 本轮成果

- `brainmirror/scept/paths.py`：新建独立跨层受控路径干预器。
- 完成从源残差边沿受控恒等桥，经过下一层实际 Attention（区分 Q/K/V 输入）或 MLP 重新计算，再隔离到最后边的路径特定干预。
- 支持 2-hop、3-hop 和预算限制的更长有序路线；中途冻结非路线分支，不误将旁路算作经过指定中间组件。
- 通过 source-token Boolean mask 约束**源边**变化；因为后续 Attention 跨 token 混合，不对目标 token 范围作不真实的保证。
- 对原始/参考进行对称但不要求数值对称的干预，记录输入效应、路径破坏、路径恢复、非目标影响、符号一致性和目标函数。
- 自动排序和有限次运行的路径搜索，使用与选择不同的参考策略回放冻结的最佳候选。
- 新增 14 个跨层数值/对照测试，总本地测试 33/33 通过。
- 保存 `runs/scca_cross_layer_smoke.json`；结果中既包含方向一致的候选，也保留效果方向冲突的测试。

## 核心正负对照

1. 手工构造的跨层残差+MLP oracle：模块重新计算的 Delta 与独立数学构造匹配。
2. 注意力 Q/K/V 输入：分别替换相应输入，输出与直接调用实际 PyTorch MHA 的对应输入结果匹配。
3. Q 投影置零：从上游来的变化不应通过 Q/K 影响 Attention 输出，但 V 路由仍可改变输出；通过。
4. 三跳中间 Q 路线与不经过它的直达旁路具有不同效应；通过。
5. 零路径变化、空 token 掩膜：输出严格保持原值；通过。
6. 非法路径和状态维度应拒绝；Hook 不泄漏；参数不变；通过。

## 真实合成 smoke（未训练模型；非科学结论）

- 固定 seed: 240216，3 层/每层 4 个 head、8 个 token 的小型 3D Transformer-VIB。
- 在候选搜索中实测 12 条路径。
- 选择参考 `channel_mean`、复核参考 `local_blur`，结果保存完整路径与分数。
- **选择参考**：输入效应约 `0.0020554`，所选路径恢复效应约 `0.00016083`，破坏效应约 `0.00013404`。
- **保留参考**：输入效应约 `0.00024447`，同一路径恢复效应约 `0.00017249`，破坏效应约 `0.00015533`。
- 预选的三跳测试具有相反效应方向，原样记录为 `direction_aligned=false`，未将其涂改为成功。

以上数值没有统计学意义，仅证明工程运行能记录实测数据。

## **仍然没有实现**

- 注意力单独 head 及 token→token 的原子边干预。
- CNN/Swin/Window-Attention/Patch-Merging 的通用跨尺度路由。
- VIB 内部随机样本路径与 Encoder–Decoder Skip 的边级路由。
- 医学上合理的 MRI 反事实参考、独立患者验证、Grad-CAM 对照实验。
- 证明找到完整或唯一的真实模型因果回路，证明优于 Grad-CAM。

## 运行

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
python -m scripts.scca_path_smoke --output runs/scca_cross_layer_smoke.json
```

正式运行前应将模块放在完整 BrainMirror 仓库中；目前工程输出全部是合成数据，不需 Kaggle 或真实患者 MRI。
