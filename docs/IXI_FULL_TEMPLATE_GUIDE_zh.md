# BrainMirror IXI 全训练队列三维解剖模板：工程运行指南

> **目标是三维结构群体模板，不是脑神经元电活动、脑功能或全脑生物物理仿真。**

## 已有真实数据
- 私有 Kaggle 数据集 `brainmirror-ixi-t1-raw`：581 份独立 IXI T1 NIfTI。
- P1 初步 QC：581/581 NIfTI header 检查通过；仅部分 voxel QC，不能代表全量合格。
- 已生成 `brainmirror_ixi_pilot5_v1` 五例的真实脑组织仿射配准候选（Dice 分别 1、0.9433、0.9533、0.9334、0.9533），未做独立三维质量签核。
- `ixi_split_draft.py` 输出的受试者级私有划分 CSV 仍是**草案**（`qc_approved=false`）。

## 自动化的内容
`ixi_full_atlas.py` 的三个命令：
1. `plan`：核对受试者级 Train/Dev/Test 清单、pilot5 泄漏约束、源文件及参考模板文件，计算需要的批次数。
2. `shard`：**只处理 Train**；每批独立完成 SynthStrip 掩膜、脑组织提取、仿射配准、配准脑掩膜 Dice 质控；生成强度累计和、平方累计和、覆盖数及受试者级私有报告。原始 MRI 始终在 `/kaggle/input`。
3. `merge`：只有全部批次文件齐全、元数据/受试者 ID/参考指纹一致时，才合并平均、标准差、覆盖率三张 **3D NIfTI**。

## Kaggle 运行方式
在此前完成试运行并有文件的私有 Notebook 中：

```python
%pip install -q numpy nibabel SimpleITK surfa
```

下载确定版本的 `scripts/ixi_full_atlas.py`，先用 `plan` 检查（示例）：

```bash
python ixi_full_atlas.py plan \
 --manifest /kaggle/working/brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv \
 --pilot-root /kaggle/working/brainmirror_ixi_pilot5_v1 --shard-size 12
```

计划输出确认后，先执行一个 batch：

```bash
python ixi_full_atlas.py shard \
 --manifest /kaggle/working/brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv \
 --pilot-root /kaggle/working/brainmirror_ixi_pilot5_v1 \
 --synthstrip-dir /kaggle/working/brainmirror_ixi_pilot \
 --shard-index 0 --shard-size 12 \
 --out /kaggle/working/brainmirror_ixi_shards/shard_000
```

每个 batch 使用唯一 `shard_NNN` 目录；**不要覆盖已成功批次**。按计划输出的总数执行其余批次。每批完成后利用 Kaggle `Save Version` 检查产物已持久保存；注意 `Save & Run All` 可能重新运行 Notebook 并与目录唯一性保护冲突，所以推荐专用参数化 Notebook 或复制独立执行单元（只运行所需批次）。后续会话通过私有 Kaggle Output / Dataset 输入恢复前批次，避免从 0 开始。

完成所有 batch 之后：

```bash
python ixi_full_atlas.py merge \
 --manifest /kaggle/working/brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv \
 --pilot-root /kaggle/working/brainmirror_ixi_pilot5_v1 \
 --shard-size 12 --shards-root /kaggle/working/brainmirror_ixi_shards \
 --out /kaggle/working/brainmirror_ixi_train_template_candidate
```

生成 `ixi_train_affine_mean.nii.gz`, `ixi_train_affine_sd.nii.gz`, `ixi_train_coverage.nii.gz`。

## 真实科研审查前提
- pilot 已观察样本强制仅 Train。其他 Dev/Test **不会**用于模板构建。
- 这套程序即使执行完，也仅产生 **single-reference affine candidate**；无法代替非线性群体模板、无偏注册、跨站点偏差及局部解剖 QC。
- `min_dice` 仅为自动异常警戒，Dice 合格并不代表配准解剖正确。
- 被拒案例必须人工审查；**不可因为方便而静默删病例并宣布通过**。
- `qc_approved=false` 的草案受试者结果最多是未审候选；重新锁定后 `--approved` 才允许进入审计签核路径，但程序输出本身仍保持 `P1_PASS=false`。
- 必须逐中心检查局部皮层、小脑、脑室配准，检查 mask 完整性及患者来源许可，最终还需要添加经过审核的脑区标签/ atlas segmentation，网站才有可点击的真实脑区。
- 真实 MRI、逐人报告、提取的脑部表面与未审模板均保存在私有 Kaggle 或同等受限空间；不向公开 GitHub/Vercel 上传。只提交源码、匿名汇总和文档。